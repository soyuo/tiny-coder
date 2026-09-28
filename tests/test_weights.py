from pathlib import Path
import io

import pytest

from tinycode.runtime.engine import InferenceRuntime
from tinycode.runtime.context import RepositoryContext
from tinycode.runtime.kv_cache import KVCacheError, KVCacheStore
from tinycode.runtime.manifest import ManifestError, ModelManifest
from tinycode.runtime.memory import MemoryLimitError, parse_memory_limit, plan_memory
from tinycode.runtime.tensor_format import TensorFormatError, pack_header, read_header
from tinycode.runtime.ops import matmul
from tinycode.runtime.tokenizer import ByteTokenizer, TokenizationError
from tinycode.runtime.attention import scaled_dot_product_attention
from tinycode.runtime.model import DecoderBlock, DecoderConfig, DecoderOnlyTransformer, DiskDecoderOnlyTransformer, _load_block_weights
from tinycode.runtime.packed_format import pack_tensors
from tinycode.runtime.generation import generate_greedy, generate_greedy_cached, generate_greedy_cached_with_cache, generate_text
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


def test_weight_store_respects_byte_budget(tmp_path: Path) -> None:
    (tmp_path / "layer_00.bin").write_bytes(b"a" * 8)
    (tmp_path / "layer_01.bin").write_bytes(b"b" * 8)

    with WeightStore(tmp_path, cache_size=2, byte_budget=8) as store:
        store.prefetch_layer(0)
        store.prefetch_layer(1)

        assert store.cached_bytes == 8
        assert store.cached_layers() == (1,)


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


def test_kv_cache_round_trips_arrays(tmp_path: Path) -> None:
    import numpy as np

    cache = KVCacheStore(tmp_path / "kv", hot_capacity=1)
    key = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    value = key + 1

    cache.put_arrays(0, key, value)
    loaded_key, loaded_value = cache.get_arrays(0)

    np.testing.assert_array_equal(loaded_key, key)
    np.testing.assert_array_equal(loaded_value, value)


def test_kv_cache_clear_removes_previous_session(tmp_path: Path) -> None:
    cache = KVCacheStore(tmp_path / "kv")
    cache.put(0, b"old")

    cache.clear()

    assert not cache.contains(0)


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


def test_memory_plan_allocates_cache_and_reserve() -> None:
    plan = plan_memory(1024, 100)

    assert plan.weights_bytes == 411
    assert plan.kv_bytes == 307
    assert plan.context_bytes == 102
    assert plan.reserve_bytes == 204
    assert plan.layer_cache == 4
    assert plan.prefetch is True


def test_memory_plan_rejects_invalid_ratios() -> None:
    with pytest.raises(MemoryLimitError, match="ratios"):
        plan_memory(1024, 100, kv_ratio=0.8, context_ratio=0.2, reserve_ratio=0.1)


def test_memory_plan_requires_room_for_one_layer() -> None:
    with pytest.raises(MemoryLimitError, match="one layer"):
        plan_memory(100, 50, kv_ratio=0.3, context_ratio=0.2, reserve_ratio=0.2)


def test_kv_cache_respects_hot_byte_budget(tmp_path: Path) -> None:
    store = KVCacheStore(tmp_path / "kv", hot_capacity=2, hot_bytes=5)

    store.put(0, b"1234")
    store.put(1, b"56")

    assert store.hot_size == 2
    assert store.hot_layers() == (1,)
    assert store.get(1) == b"56"
    assert store.hot_size == 2


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


def test_decoder_block_rejects_invalid_projection_shape(tmp_path: Path) -> None:
    import numpy as np

    packed = pack_tensors([("q_proj", "float32", (1, 2), np.zeros((1, 2), dtype=np.float32).tobytes())])
    write_layer(tmp_path, 0, packed)
    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)

    with WeightStore(tmp_path) as store:
        with store.load_packed_layer(0) as layer:
            with pytest.raises(ValueError, match="q_proj has shape"):
                _load_block_weights(config, layer.packed_layer())


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


def test_disk_decoder_cleans_up_after_tensor_load_failure(tmp_path: Path, monkeypatch) -> None:
    import json
    import numpy as np
    import tinycode.runtime.model as model_module

    config = {"num_layers": 1, "vocab_size": 8, "hidden_size": 4, "intermediate_size": 6, "num_heads": 2}
    (tmp_path / "model.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "embedding.bin").write_bytes(
        pack_header("float32", (8, 4)) + np.ones((8, 4), dtype=np.float32).tobytes()
    )
    original = model_module._load_tensor_file
    calls = 0

    def fail_on_second_load(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("cannot load norm")
        return original(path)

    monkeypatch.setattr(model_module, "_load_tensor_file", fail_on_second_load)
    with pytest.raises(OSError, match="cannot load norm"):
        DiskDecoderOnlyTransformer.from_model_dir(tmp_path)


def test_disk_decoder_cached_forward_keeps_previous_kv(tmp_path: Path) -> None:
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
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    (tmp_path / "layer_00.bin").write_bytes(
        pack_tensors([(name, "float32", array.shape, array.tobytes()) for name, array in arrays.items()])
    )
    store = KVCacheStore(tmp_path / "kv")

    with DiskDecoderOnlyTransformer.from_model_dir(tmp_path) as model:
        cache = model.new_cache(store)
        model.forward_cached(np.array([1], dtype=np.int64), cache)
        model.forward_cached(np.array([2], dtype=np.int64), cache)

    key, _ = store.get_arrays(0)
    assert key.shape[1] == 2


def test_greedy_generation_stops_at_eos() -> None:
    import numpy as np

    class FakeModel:
        def __call__(self, token_ids: object) -> object:
            return np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(token_ids), 1))

    assert generate_greedy(FakeModel(), [0], max_new_tokens=3) == [0, 1, 1, 1]
    assert generate_greedy(FakeModel(), [0], max_new_tokens=3, eos_token_id=1) == [0, 1]


def test_cached_generation_reuses_decoder_states() -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config, np.ones((8, 4), dtype=np.float32), [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32),
    )

    result = generate_greedy_cached(model, [1, 2], max_new_tokens=2)

    assert result == [1, 2, 0, 0]


def test_cached_generation_appends_to_reusable_kv_buffer() -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config, np.ones((8, 4), dtype=np.float32), [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32),
    )
    cache = model.new_cache()

    model.forward_cached(np.array([1], dtype=np.int64), cache)
    buffer = cache.key_buffers[0]
    model.forward_cached(np.array([2], dtype=np.int64), cache)

    assert cache.key_buffers[0] is buffer
    assert cache.keys[0].shape[1] == 2


def test_cached_logits_match_full_prefix_logits() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        name: rng.normal(size=shape).astype(np.float32)
        for name, shape in {
            "q_proj": (4, 4), "k_proj": (4, 4), "v_proj": (4, 4), "o_proj": (4, 4),
            "gate_proj": (6, 4), "up_proj": (6, 4), "down_proj": (4, 6),
            "input_norm": (4,), "post_norm": (4,),
        }.items()
    }
    model = DecoderOnlyTransformer(
        config, rng.normal(size=(8, 4)).astype(np.float32), [DecoderBlock(config, weights)],
        rng.normal(size=4).astype(np.float32), rng.normal(size=(8, 4)).astype(np.float32),
    )
    prompt = np.array([1, 2, 3], dtype=np.int64)
    cache = model.new_cache()

    cached_prompt = model.forward_cached(prompt, cache)
    full_prompt = model(prompt)
    cached_next = model.forward_cached(np.array([4], dtype=np.int64), cache)
    full_next = model(np.array([1, 2, 3, 4], dtype=np.int64))

    np.testing.assert_allclose(cached_prompt, full_prompt, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cached_next, full_next[-1:], rtol=1e-5, atol=1e-5)


def test_disk_cached_logits_match_in_memory_cache(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        name: rng.normal(size=shape).astype(np.float32)
        for name, shape in {
            "q_proj": (4, 4), "k_proj": (4, 4), "v_proj": (4, 4), "o_proj": (4, 4),
            "gate_proj": (6, 4), "up_proj": (6, 4), "down_proj": (4, 6),
            "input_norm": (4,), "post_norm": (4,),
        }.items()
    }
    model = DecoderOnlyTransformer(
        config, rng.normal(size=(8, 4)).astype(np.float32), [DecoderBlock(config, weights)],
        rng.normal(size=4).astype(np.float32), rng.normal(size=(8, 4)).astype(np.float32),
    )
    prompt = np.array([1, 2, 3], dtype=np.int64)
    memory_cache = model.new_cache()
    disk_cache = model.new_cache(KVCacheStore(tmp_path / "kv", hot_capacity=1))

    memory_prompt = model.forward_cached(prompt, memory_cache)
    disk_prompt = model.forward_cached(prompt, disk_cache)
    memory_next = model.forward_cached(np.array([4], dtype=np.int64), memory_cache)
    disk_next = model.forward_cached(np.array([4], dtype=np.int64), disk_cache)

    np.testing.assert_allclose(disk_prompt, memory_prompt, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(disk_next, memory_next, rtol=1e-5, atol=1e-5)


def test_cached_generation_can_spill_kv_to_disk(tmp_path: Path) -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config, np.ones((8, 4), dtype=np.float32), [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32),
    )
    store = KVCacheStore(tmp_path / "kv", hot_capacity=1)
    cache = model.new_cache(store)

    model.forward_cached(np.array([1, 2], dtype=np.int64), cache)

    assert cache.keys == [None]
    assert store.contains(0)


def test_cached_generation_accepts_caller_cache(tmp_path: Path) -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config, np.ones((8, 4), dtype=np.float32), [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32),
    )
    cache = model.new_cache(KVCacheStore(tmp_path / "kv"))

    assert generate_greedy_cached_with_cache(model, [1], 1, cache) == [1, 0]


def test_cached_generation_rejects_invalid_disk_kv_shape(tmp_path: Path) -> None:
    import numpy as np

    config = DecoderConfig(vocab_size=8, hidden_size=4, intermediate_size=6, num_layers=1, num_heads=2)
    weights = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    model = DecoderOnlyTransformer(
        config, np.ones((8, 4), dtype=np.float32), [DecoderBlock(config, weights)],
        np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32),
    )
    store = KVCacheStore(tmp_path / "kv")
    cache = model.new_cache(store)
    store.put_arrays(0, np.zeros((1, 1, 4), dtype=np.float32), np.zeros((1, 1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="cached key/value shape"):
        model.forward_cached(np.array([1], dtype=np.int64), cache)


def test_byte_tokenizer_validates_model_vocabulary() -> None:
    tokenizer = ByteTokenizer()
    tokenizer.validate_vocab_size(256)
    with pytest.raises(TokenizationError):
        tokenizer.validate_vocab_size(128)
