from pathlib import Path
import io

import pytest

from tinycode.runtime.engine import InferenceRuntime
from tinycode.runtime.manifest import ManifestError, ModelManifest
from tinycode.runtime.tensor_format import TensorFormatError, pack_header, read_header
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
        store.load_layer(0)
        store.load_layer(1)
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
        layer = store.load_tensor_layer(0)
        assert layer.tensor_header().data_size == len(payload)


def test_weight_store_rejects_truncated_tensor_layer(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, pack_header("float32", (2, 2)) + b"\x00")

    with WeightStore(tmp_path) as store:
        with pytest.raises(TensorFormatError, match="truncated"):
            store.load_tensor_layer(0)


def test_weight_store_prefetches_on_worker(tmp_path: Path) -> None:
    write_layer(tmp_path, 0, b"zero")

    with WeightStore(tmp_path) as store:
        layer = store.prefetch_layer_async(0).result(timeout=2)
        assert layer.read() == b"zero"
        assert store.cached_layers() == (0,)
