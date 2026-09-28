from pathlib import Path

import numpy as np

from tinycode.cli import build_parser, run_command
from tinycode.runtime.packed_format import pack_tensors
from tinycode.runtime.tensor_format import pack_header


def write_decoder_fixture(model_dir: Path) -> None:
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.json").write_text(
        '{"num_layers": 1, "vocab_size": 256, "hidden_size": 4, '
        '"intermediate_size": 6, "num_heads": 2}',
        encoding="utf-8",
    )
    embedding = np.ones((256, 4), dtype=np.float32)
    norm = np.ones(4, dtype=np.float32)
    lm_head = np.zeros((256, 4), dtype=np.float32)
    lm_head[0] = 1.0
    for name, array in (("embedding.bin", embedding), ("norm.bin", norm), ("lm_head.bin", lm_head)):
        (model_dir / name).write_bytes(pack_header("float32", array.shape) + array.tobytes())
    arrays = {
        "q_proj": np.eye(4, dtype=np.float32), "k_proj": np.eye(4, dtype=np.float32),
        "v_proj": np.eye(4, dtype=np.float32), "o_proj": np.eye(4, dtype=np.float32),
        "gate_proj": np.ones((6, 4), dtype=np.float32), "up_proj": np.ones((6, 4), dtype=np.float32),
        "down_proj": np.ones((4, 6), dtype=np.float32), "input_norm": np.ones(4, dtype=np.float32),
        "post_norm": np.ones(4, dtype=np.float32),
    }
    (model_dir / "layer_00.bin").write_bytes(
        pack_tensors([(name, "float32", array.shape, array.tobytes()) for name, array in arrays.items()])
    )


def test_run_command_without_prompt_keeps_runtime_ready(tmp_path: Path, capsys) -> None:
    (tmp_path / "model.json").write_text('{"num_layers": 1}', encoding="utf-8")
    args = build_parser().parse_args(["run", "--model", str(tmp_path), "--layer-cache", "1"])

    assert run_command(args) == 0
    assert "CPU runtime ready" in capsys.readouterr().out


def test_run_command_rejects_negative_generation_length(tmp_path: Path, capsys) -> None:
    (tmp_path / "model.json").write_text('{"num_layers": 1}', encoding="utf-8")
    args = build_parser().parse_args(["run", "--model", str(tmp_path), "--max-new-tokens", "-1"])

    assert run_command(args) == 2
    assert "max new tokens" in capsys.readouterr().out


def test_run_command_plans_layer_cache_from_memory_limit(tmp_path: Path, capsys) -> None:
    (tmp_path / "model.json").write_text('{"num_layers": 1}', encoding="utf-8")
    (tmp_path / "layer_00.bin").write_bytes(b"x" * 100)
    args = build_parser().parse_args(["run", "--model", str(tmp_path), "--memory-limit", "1K"])

    assert run_command(args) == 0
    output = capsys.readouterr().out
    assert "layer cache: 4" in output
    assert "prefetch: enabled" in output


def test_run_command_generates_with_disk_decoder(tmp_path: Path, capsys) -> None:
    write_decoder_fixture(tmp_path)
    args = build_parser().parse_args(
        ["run", "--model", str(tmp_path), "--memory-limit", "16K", "--prompt", "a", "--max-new-tokens", "1"]
    )

    assert run_command(args) == 0
    output = capsys.readouterr().out
    assert "prefetch: enabled" in output
    assert "KV cache: disk" in output


def test_run_command_reports_selected_repository_context(tmp_path: Path, capsys) -> None:
    model_dir = tmp_path / "model"
    repository = tmp_path / "repo"
    write_decoder_fixture(model_dir)
    repository.mkdir()
    (repository / "auth.py").write_text("def refresh_token(): pass\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "run", "--model", str(model_dir), "--memory-limit", "16K",
            "--repository", str(repository), "--prompt", "refresh token", "--max-new-tokens", "0",
        ]
    )

    assert run_command(args) == 0
    assert "context files: 1" in capsys.readouterr().out
