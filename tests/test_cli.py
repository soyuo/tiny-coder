from pathlib import Path

from tinycode.cli import build_parser, run_command


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
