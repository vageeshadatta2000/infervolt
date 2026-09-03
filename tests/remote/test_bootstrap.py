"""The bootstrap script is a string, so every claim about it is a unit test.

Nothing here runs the script: the point of building it as a pure function is that its
content -- which commit it installs, which secrets it carries, what it refuses to redo --
can be asserted on a laptop with no GPU and no network.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from infervolt.remote import bootstrap as bootstrap_module
from infervolt.remote.bootstrap import (
    BOOTSTRAP_OK,
    DEFAULT_VLLM_VERSION,
    REPO_URL,
    WORK_MARKER,
    bootstrap_script,
    controller_sha,
    optimize_command,
    passthrough_env,
    stamp,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_the_script_installs_the_exact_commit_the_controller_is_running() -> None:
    script = bootstrap_script(ref=SHA)
    assert f"infervolt @ git+{REPO_URL}@{SHA}" in script
    assert "uv venv --python 3.12" in script
    assert 'uv pip install --python "$WORK/venv/bin/python"' in script


def test_uv_is_installed_only_when_missing_and_put_on_the_path() -> None:
    script = bootstrap_script(ref=SHA)
    assert "command -v uv" in script
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in script
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script


def test_the_work_dir_falls_back_to_home_when_workspace_is_not_writable() -> None:
    script = bootstrap_script(ref=SHA)
    assert "/workspace/infervolt" in script
    assert 'WORK="$HOME/infervolt"' in script
    # The controller has to learn which of the two won, because scp cannot expand $WORK.
    assert f'echo "{WORK_MARKER}$WORK"' in script


def test_the_venv_engine_install_pins_the_vllm_version() -> None:
    script = bootstrap_script(ref=SHA)
    assert f"vllm=={DEFAULT_VLLM_VERSION}" in script
    pinned = bootstrap_script(ref=SHA, vllm_version="0.12.3")
    assert "vllm==0.12.3" in pinned and f"vllm=={DEFAULT_VLLM_VERSION}" not in pinned


def test_the_docker_engine_install_pulls_an_image_and_installs_no_vllm() -> None:
    script = bootstrap_script(ref=SHA, engine_install="docker", image="vllm/vllm-openai:v0.11.0")
    assert "docker pull 'vllm/vllm-openai:v0.11.0'" in script
    assert "vllm==" not in script


def test_docker_without_an_image_is_a_usage_error() -> None:
    with pytest.raises(ValueError, match="image"):
        bootstrap_script(ref=SHA, engine_install="docker")


def test_hf_home_lives_on_the_box_not_in_the_root_filesystem() -> None:
    assert 'export HF_HOME="$WORK/hf"' in bootstrap_script(ref=SHA)


def test_only_variables_that_are_actually_set_are_forwarded() -> None:
    script = bootstrap_script(ref=SHA, env={"HF_TOKEN": "hf_secret", "ANTHROPIC_API_KEY": "sk-a"})
    assert "export HF_TOKEN='hf_secret'" in script
    assert "export ANTHROPIC_API_KEY='sk-a'" in script
    bare = bootstrap_script(ref=SHA)
    assert "HF_TOKEN" not in bare and "ANTHROPIC_API_KEY" not in bare


def test_forwarded_values_are_quoted_so_a_key_with_shell_metacharacters_survives() -> None:
    script = bootstrap_script(ref=SHA, env={"HF_TOKEN": "a b'c;rm -rf /"})
    assert "rm -rf /" in script
    # Inside a single-quoted word, so it is data rather than a second command.
    assert "export HF_TOKEN='a b'\"'\"'c;rm -rf /'" in script


def test_the_secrets_go_to_a_file_the_optimize_command_can_source() -> None:
    script = bootstrap_script(ref=SHA, env={"HF_TOKEN": "hf_secret"})
    assert "umask 077" in script
    assert '"$WORK/env.sh"' in script
    assert "/w/env.sh" in optimize_command("/w", ["--engine", "vllm"])


def test_passthrough_reads_the_controller_environment_not_files() -> None:
    env = passthrough_env(
        {
            "HF_TOKEN": "hf_x",
            "ANTHROPIC_API_KEY": "sk-a",
            "INFERVOLT_ANTHROPIC_MODEL": "claude-opus-5",
            "INFERVOLT_HOME": "/Users/someone/.infervolt",
            "TNR_API_TOKEN": "tnr_x",
            "PATH": "/usr/bin",
        }
    )
    assert env == {
        "HF_TOKEN": "hf_x",
        "ANTHROPIC_API_KEY": "sk-a",
        "INFERVOLT_ANTHROPIC_MODEL": "claude-opus-5",
    }
    # INFERVOLT_HOME would point the box at a path on the controller; the runner passes
    # --home explicitly and the provider token is the controller's business alone.


def test_passthrough_ignores_empty_values() -> None:
    assert passthrough_env({"HF_TOKEN": "", "ANTHROPIC_API_KEY": "sk-a"}) == {
        "ANTHROPIC_API_KEY": "sk-a"
    }


def test_rerunning_the_same_bootstrap_is_a_fast_path_not_a_reinstall() -> None:
    script = bootstrap_script(ref=SHA)
    marker = stamp(ref=SHA, engine_install="venv", vllm_version=DEFAULT_VLLM_VERSION, image=None)
    assert marker in script
    assert '"$WORK/.bootstrap-ok"' in script
    # The fast path still announces success, or the runner would think it failed.
    ok_at = script.index(BOOTSTRAP_OK)
    assert "exit 0" in script[ok_at : ok_at + 40]


def test_the_stamp_changes_with_everything_that_changes_the_box() -> None:
    base = stamp(ref=SHA, engine_install="venv", vllm_version="0.11.0", image=None)
    assert base != stamp(ref="other", engine_install="venv", vllm_version="0.11.0", image=None)
    assert base != stamp(ref=SHA, engine_install="venv", vllm_version="0.12.0", image=None)
    assert base != stamp(ref=SHA, engine_install="docker", vllm_version="0.11.0", image="i")


def test_the_script_stops_at_the_first_failure() -> None:
    assert bootstrap_script(ref=SHA).startswith("set -euo pipefail")


def test_a_ref_that_is_not_a_git_ref_is_refused_before_it_reaches_a_shell() -> None:
    with pytest.raises(ValueError, match="ref"):
        bootstrap_script(ref="main; rm -rf /")


def test_the_optimize_command_runs_the_installed_infervolt_against_remote_state() -> None:
    cmd = optimize_command("/workspace/infervolt", ["--engine", "vllm", "--model", "Qwen/Qwen3-8B"])
    assert "/workspace/infervolt/venv/bin/infervolt optimize" in cmd
    assert "--home '/workspace/infervolt/state'" in cmd
    assert "--engine vllm --model Qwen/Qwen3-8B" in cmd
    # exec, so that the cost cap's pkill has a process whose cmdline it can match.
    assert "exec " in cmd


def test_the_optimize_command_quotes_arguments() -> None:
    args = ["--slo", "ttft=500ms,itl=30ms", "--baseline", "x=a b", "--note", "it's fine"]
    line = optimize_command("/w", args).splitlines()[-1]
    # Whatever the quoting, the remote shell must hand infervolt exactly these words back.
    assert shlex.split(line) == [
        "exec",
        "/w/venv/bin/infervolt",
        "optimize",
        "--home",
        "/w/state",
        *args,
    ]


def test_controller_sha_reads_the_current_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x")
    subprocess.run(["git", "add", "f"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "c"], cwd=tmp_path, check=True)
    sha = controller_sha(tmp_path)
    assert sha is not None and len(sha) == 40


def test_controller_sha_asks_about_this_package_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Running the CLI from inside somebody else's repository must not ship their HEAD.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    assert controller_sha() == controller_sha(Path(bootstrap_module.__file__).parent)


def test_controller_sha_is_none_outside_a_checkout(tmp_path: Path) -> None:
    # An infervolt installed from a wheel has no HEAD to pin; the caller must say --ref.
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert controller_sha(plain) is None
    assert controller_sha(tmp_path / "does-not-exist") is None


def test_read_dotenv_and_passthrough_prefers_process_env(tmp_path, monkeypatch) -> None:
    from infervolt.remote.bootstrap import passthrough_env, read_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "# keys\n"
        'export ANTHROPIC_API_KEY="from-file"\n'
        "HF_TOKEN=hf-x\n"
        "INFERVOLT_THUNDER_API_TOKEN=nope\n"
    )
    assert read_dotenv(env) == {
        "ANTHROPIC_API_KEY": "from-file",
        "HF_TOKEN": "hf-x",
        "INFERVOLT_THUNDER_API_TOKEN": "nope",
    }
    assert read_dotenv(tmp_path / "missing") == {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = passthrough_env()
    assert out["ANTHROPIC_API_KEY"] == "from-env" and out["HF_TOKEN"] == "hf-x"
    assert "INFERVOLT_THUNDER_API_TOKEN" not in out
