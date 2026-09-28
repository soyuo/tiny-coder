from pathlib import Path
import io

import pytest

from tinycode.runtime.engine import InferenceRuntime
from tinycode.runtime.context import RepositoryContext
from tinycode.runtime.kv_cache import KVCacheError, KVCacheStore
from tinycode.runtime.manifest import ManifestError, ModelManifest
from tinycode.runtime.memory import parse_memory_limit
from tinycode.runtime.tensor_format import TensorFormatError, pack_header, read_header
from tinycode.runtime.ops import matmul
from tinycode.runtime.tokenizer import ByteTokenizer, TokenizationError
from tinycode.runtime.attention import scaled_dot_product_attention
from tinycode.runtime.model import DecoderBlock, DecoderConfig, DecoderOnlyTransformer, DiskDecoderOnlyTransformer
from tinycode.runtime.packed_format import pack_tensors
from tinycode.runtime.generation import generate_greedy, generate_text
from tinycode.runtime.weights import WeightStore, WeightStoreError


def write_layer(model_dir: Path, index: int, payload: bytes) -> None:
    model_dir.mkdir(exist_ok=True)
    (model_dir / f"layer_{index:02d}.bin").write_bytes(payload)


def test_load_layer_maps_bytes_and_unload_releases_cache(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, b"layer-zero")

    with WeightStore(tmp_path) as store:
        layer = store.load_layer(0)
        assert layer.read() == b"layer-zero"
        assert store.cached_layers() == (0,)
        store.unload_layer(0)
        assert store.cached_layers() == ()


def test_cache_evicts_least_recently_used_layer(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, b"zero")
    write_layer(tmp_path, 1, b"one")

    with WeightStore(tmp_path, cache_size=1) as store:
        store.prefetch_layer(0)
        store.prefetch_layer(1)
        assert store.cached_layers() == (1,)


def test_missing_layer_is_reported(tmp_path: Path) -> None:
    with WeightStore(tmp_path) as store:
        with pytest.raises(WeightStoreError, match="does not exist"):
            store.load_layer(0)


def test_runtime_processes_layers_in_order_and_releases_each(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, b"A")
    write_layer(tmp_path, 1, b"B")
    seen: list[bytes] = []

    def execute(state: str, layer: object) -> str:
        seen.append(layer.read())
        return state + layer.read().decode()

    with InferenceRuntime(tmp_path, layer_cache=1) as runtime:
        result = runtime.run_layers("", [0, 1], execute)
        assert result == "AB"
        assert seen == [b"A", b"B"]
        assert runtime.weights.cached_layers() == ()


def test_manifest_controls_layer_count_and_filename_width(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text(
        '{"num_layers": 3, "layer_digits": 3}', encoding="utf-8"
    )
    manifest = ModelManifest.load(tmp_path)
    write_layer(tmp_path, 2, b"two")

    with WeightStore(tmp_path, manifest=manifest) as store:
        assert store.layer_count() == 3
        assert store.layer_path(2).name == "layer_002.bin"


def test_manifest_rejects_invalid_layer_count(tmp_path: Path) -> None:
    (tmp_path / "model.json").write_text('{"num_layers": 0}', encoding="utf-8")

    with pytest.raises(ManifestError, match="positive integer"):
        ModelManifest.load(tmp_path)


def test_tensor_header_round_trip() -> None:
    header = pack_header("float16", (2, 3))
    parsed = read_header(io.BytesIO(header))

    assert parsed.dtype == "float16"
    assert parsed.shape == (2, 3)
    assert parsed.data_offset == 16
    assert parsed.data_size == 12


def test_tensor_header_rejects_bad_magic() -> None:
    with pytest.raises(TensorFormatError, match="unsupported tensor format"):
        read_header(io.BytesIO(b"BAD!\x01\x01\x01\x00"))


def test_weight_store_validates_mapped_tensor_layer(tmp_path: Path) -> None:
    payload = b"\x00" * 12
    write_layer(tmp_path, 0, pack_header("float16", (2, 3)) + payload)

    with WeightStore(tmp_path) as store:
        with store.load_tensor_layer(0) as layer:
            assert layer.tensor_header().data_size == len(payload)


def test_weight_store_rejects_truncated_tensor_layer(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, pack_header("float32", (2, 2)) + b"\x00")

    with WeightStore(tmp_path) as store:
        with pytest.raises(TensorFormatError, match="truncated"):
            store.load_tensor_layer(0)


def test_weight_store_prefetches_on_worker(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, b"zero")

    with WeightStore(tmp_path) as store:
        assert store.prefetch_layer_async(0).result(timeout=2) is None
        with store.hold_layer(0) as layer:
            assert layer.read() == b"zero"
        assert store.cached_layers() == ()


def test_kv_cache_spills_cold_entries_to_disk(tmp_path: Path) -> None:
    cache = KVCacheStore(tmp_path / "kv", hot_capacity=1)
    cache.put(0, b"zero")
    cache.put(1, b"one")

    assert cache.hot_layers() == (1,)
    assert (tmp_path / "kv" / "layer_00.cache").read_bytes() == b"zero"
    assert cache.get(0) == b"zero"
    assert cache.hot_layers() == (0,)


def test_kv_cache_reports_missing_entry(tmp_path: Path) -> None:
    cache = KVCacheStore(tmp_path / "kv")

    with pytest.raises(KVCacheError, match="does not exist"):
        cache.get(0)


def test_repository_context_returns_relevant_files_with_limits(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def refresh_token():\n    pass\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("def render_home():\n    pass\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "auth.py").write_text("refresh_token", encoding="utf-8")

    results = RepositoryContext(tmp_path, max_files=1, max_bytes=20).search("refresh token")

    assert len(results) == 1
    assert results[0].path.name == "auth.py"
    assert len(results[0].text.encode("utf-8")) <= 20


def test_memory_limit_parser() -> None:
    assert parse_memory_limit("512M") == 512 * 1024**2
    assert parse_memory_limit("1G") == 1024**3


def test_cpu_tensor_layer_execution(tmp_path: Path) -> None:
    payload = b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
    write_layer(tmp_path, 0, pack_header("float32", (2, 2)) + payload)

    with InferenceRuntime(tmp_path) as runtime:
        result = runtime.run_tensor_layers(
            [[1.0, 2.0]],
            [0],
            lambda state, weights: matmul(state, weights),
        )

    assert result.tolist() == [[7.0, 10.0]]


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "def 안녕():\n    pass"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_rejects_invalid_token() -> None:
    with pytest.raises(TokenizationError):
        ByteTokenizer().decode([256])


def test_scaled_dot_product_attention_returns_finite_values() -> None:
    result = scaled_dot_product_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[2.0], [4.0]])

    assert result.shape == (1, 1)
    assert result[0, 0] > 2.0
    assert result[0, 0] < 4.0


def test_decoder_only_transformer_returns_next_token_logits() -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32),
        "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32),
        "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32),
        "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32),
        "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config,
        np.ones((8, 4), dtype=np.float32),
        [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32),
        np.ones((8, 4), dtype=np.float32),
    )

    logits = model(np.array([1, 2, 3]))

    assert logits.shape == (3, 8)
    assert np.isfinite(logits).all()


def test_packed_layer_reads_named_tensors(tmp_path: Path) -> None:
    packed = pack_tensors([
        ("q_proj", "float32", (1, 2), b"\x00\x00\x80?\x00\x00\x00@"),
        ("bias", "float32", (2,), b"\x00\x00@@\x00\x00\x80@"),
    ])
    write_layer(tmp_path, 0, packed)

    with WeightStore(tmp_path) as store:
        with store.load_packed_layer(0) as layer:
            assert layer.packed_layer().tensors["q_proj"].shape == (1, 2)
            assert layer.packed_layer().tensor_view("bias").to_numpy().tolist() == [3.0, 4.0]


def test_disk_decoder_loads_one_packed_layer_at_a_time(tmp_path: Path) -> None:
    import json
    import numpy as np

    config = {"num_layers": 1, "vocab_size": 8, "hidden_size": 4, "intermediate_size": 6, "num_heads": 2}
    (tmp_path / "model.json").write_text(json.dumps(config), encoding="utf-8")
    for name, array in {
        "embedding.bin": np.ones((8, 4), dtype=np.float32),
        "norm.bin": np.ones(4, dtype=np.float32),
        "lm_head.bin": np.ones((8, 4), dtype=np.float32),
    }.items():
        (tmp_path / name).write_bytes(pack_header("float32", array.shape) + array.tobytes())
    arrays = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32), "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32), "down_proj": np.ones((4, 6), dtype=np.float32),
        "input_norm": np.ones(4, dtype=np.float32), "post_norm": np.ones(4, dtype=np.float32),
    }
    (tmp_path / "layer_00.bin").write_bytes(pack_tensors([(name, "float32", array.shape, array.tobytes()) for name, array in arrays.items()]))

    with DiskDecoderOnlyTransformer.from_model_dir(tmp_path) as model:
        logits = model(np.array([1, 2]))

    assert logits.shape == (2, 8)


def test_greedy_generation_stops_at_eos() -> None:
    import numpy as np

    class FakeModel:
        def __call__(self, token_ids: object) -> object:
            return np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(token_ids), 1))

    assert generate_greedy(FakeModel(), [0], max_new_tokens=3) == [0, 1, 1, 1]
    assert generate_greedy(FakeModel(), [0], max_new_tokens=3, eos_token_id=1) == [0, 1]


def test_byte_tokenizer_validates_model_vocabulary() -> None:
    tokenizer = ByteTokenizer()
    tokenizer.validate_vocab_size(256)
    with pytest.raises(TokenizationError):
        tokenizer.validate_vocab_size(128)
