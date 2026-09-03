"""The script that turns a bare rented box into one that can run ``infervolt optimize``.

It is built as a string by a pure function on purpose. A bootstrap that only exists as
side effects of an SSH session can be tested by renting a GPU; a bootstrap that is a value
can be tested on a laptop, diffed in review, and printed when someone asks what exactly we
ran on the machine they are paying for.

Three properties the script must have, and the reasons they are not obvious:

* **It pins a commit, not a branch.** ``git+...@<sha>`` where ``<sha>`` is the controller's
  own ``HEAD``, so the code that produced a recipe is the code that was reviewed. A branch
  name would let the box run something the controller never saw.
* **It is idempotent.** ``--keep`` exists so a second trial does not pay for another
  install; re-running the same bootstrap therefore checks a stamp and exits in under a
  second rather than reinstalling vLLM.
* **Secrets come from the controller's environment and nowhere else.** No credential file
  is read, no ``.env`` is shipped, and only names that are actually set are forwarded --
  an unset ``HF_TOKEN`` must not become ``HF_TOKEN=`` on the box, where it would shadow a
  working one.

The script assumes bash (``pipefail``), which every image these providers offer runs as
the login shell of the account we connect as.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

REPO_URL = "https://github.com/vageeshadatta2000/infervolt"
DEFAULT_VLLM_VERSION = "0.11.0"

BOOTSTRAP_OK = "INFERVOLT_BOOTSTRAP_OK"
"""Printed as the last line of a successful bootstrap.

The exit code alone is not enough: an ``ssh`` that dies mid-install can still hand back
a zero from something further up the pipe, and the runner must not start a paid run on a
box where ``uv pip install`` never finished."""

WORK_MARKER = "INFERVOLT_WORK="
"""Printed once the work directory is chosen, because ``scp`` cannot expand ``$WORK``."""

EngineInstall = Literal["venv", "docker"]

PRIMARY_WORK_DIR = "/workspace/infervolt"
FALLBACK_WORK_DIR = "$HOME/infervolt"

PASSTHROUGH_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY")
PASSTHROUGH_PREFIX = "INFERVOLT_"
PASSTHROUGH_DENY = frozenset(
    {
        # Points at a directory on the controller; the runner passes --home itself.
        "INFERVOLT_HOME",
        # A recorded-LLM cassette is a local path, and replaying one on the box would
        # silently turn a real run into a fake one.
        "INFERVOLT_LLM_CASSETTE",
        # The controller rents the boxes. The box has no reason to be able to rent more.
        "INFERVOLT_THUNDER_API_TOKEN",
        "INFERVOLT_RUNPOD_API_KEY",
        "INFERVOLT_PRIME_API_KEY",
    }
)

_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_WORK_RE = re.compile(r"^[A-Za-z0-9_./~-]+$")


def controller_sha(cwd: Path | None = None) -> str | None:
    """The commit this controller is running, or ``None`` if it is not a git checkout.

    ``None`` rather than a guess: an infervolt installed from a wheel has no HEAD, and
    defaulting to ``main`` would ship the box something nobody reviewed. The caller turns
    it into "pass --ref".

    The default directory is this package's own, not the process's working directory: the
    question is "which infervolt is running", and someone who invoked the CLI from inside
    an unrelated checkout must not silently ship that repository's HEAD.
    """
    where = Path(__file__).resolve().parent if cwd is None else Path(cwd)
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(where),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and _REF_RE.match(sha) else None


def passthrough_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The controller variables worth having on the box, and only those.

    Two rules, both about blast radius: a variable that is not set is not forwarded (so
    the box's own configuration wins over our absence of one), and the credentials that
    rent machines stay on the controller -- the box needs a model and an LLM key, not the
    ability to provision more of itself.
    """
    source = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for name, value in source.items():
        if not value:
            continue
        if name in PASSTHROUGH_DENY:
            continue
        if name in PASSTHROUGH_NAMES or name.startswith(PASSTHROUGH_PREFIX):
            out[name] = value
    return out


def stamp(
    *,
    ref: str,
    engine_install: EngineInstall = "venv",
    vllm_version: str = DEFAULT_VLLM_VERSION,
    image: str | None = None,
) -> str:
    """A short digest of everything that changes what gets installed.

    The box stores it after a successful bootstrap; a later run with the same inputs skips
    straight to the end. Anything that would change a byte on disk -- the commit, the
    engine install mode, the vLLM pin, the image -- has to change this, or ``--keep`` would
    quietly run yesterday's code.
    """
    material = "\0".join([ref, engine_install, vllm_version, image or ""])
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _check_ref(ref: str) -> str:
    if not _REF_RE.match(ref):
        raise ValueError(f"unusable git ref {ref!r}: expected a sha, tag or branch name")
    return ref


def _check_work(work: str) -> str:
    if not _WORK_RE.match(work):
        raise ValueError(f"unusable remote work dir {work!r}")
    return work


def _q(value: str) -> str:
    """Single-quote unconditionally, the way ``shlex.quote`` does when it must.

    Unconditionally, because the strings that go through here are secrets, image tags and
    package specs: a reviewer reading the script should be able to see where a value ends
    without first working out whether this particular one happened to need quoting.
    """
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _exports(env: Mapping[str, str]) -> str:
    return "\n".join(f"export {name}={_q(env[name])}" for name in sorted(env))


def bootstrap_script(
    *,
    ref: str,
    engine_install: EngineInstall = "venv",
    vllm_version: str = DEFAULT_VLLM_VERSION,
    image: str | None = None,
    env: Mapping[str, str] | None = None,
    repo_url: str = REPO_URL,
) -> str:
    """Build the whole remote bootstrap as one shell script.

    ``env`` is written to ``$WORK/env.sh`` under ``umask 077`` rather than being re-sent
    with every later command: the optimize command sources it, so a key crosses the wire
    once instead of once per invocation.
    """
    _check_ref(ref)
    marker = stamp(ref=ref, engine_install=engine_install, vllm_version=vllm_version, image=image)
    exports = _exports(env or {})
    if engine_install == "docker":
        if not image:
            raise ValueError("engine_install='docker' needs an image to pull")
        engine_step = f"docker pull {_q(image)}"
    else:
        pin = _q(f"vllm=={vllm_version}")
        engine_step = f'uv pip install --python "$WORK/venv/bin/python" {pin}'
    spec = _q(f"infervolt @ git+{repo_url}@{ref}")
    return f"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# /workspace is the big disk on most providers and absent on some; $HOME always exists but
# is sometimes a 20GB root partition, so it is the fallback rather than the default.
if mkdir -p {PRIMARY_WORK_DIR} 2>/dev/null && [ -w {PRIMARY_WORK_DIR} ]; then
  WORK={PRIMARY_WORK_DIR}
else
  WORK="{FALLBACK_WORK_DIR}"
  mkdir -p "$WORK"
fi
export WORK
echo "{WORK_MARKER}$WORK"

export HF_HOME="$WORK/hf"
mkdir -p "$HF_HOME" "$WORK/state"

# Written every time, before the fast path: a kept instance must pick up a key that was
# rotated on the controller without paying for a reinstall to get it.
umask 077
echo "export HF_HOME=\\"$WORK/hf\\"" > "$WORK/env.sh"
cat >> "$WORK/env.sh" <<'INFERVOLT_ENV'
{exports}
INFERVOLT_ENV

STAMP={marker}
if [ -f "$WORK/.bootstrap-ok" ] && [ "$(cat "$WORK/.bootstrap-ok")" = "$STAMP" ]; then
  echo "infervolt: bootstrap $STAMP already installed"
  echo "{BOOTSTRAP_OK}"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

[ -x "$WORK/venv/bin/python" ] || uv venv --python 3.12 "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" {spec}
{engine_step}
"$WORK/venv/bin/infervolt" --version

echo "$STAMP" > "$WORK/.bootstrap-ok"
echo "{BOOTSTRAP_OK}"
"""


def optimize_command(work: str, args: Sequence[str]) -> str:
    """The remote ``infervolt optimize`` invocation, rooted in the box's own state dir.

    ``exec`` matters: the cost cap kills the run with ``pkill -f "infervolt optimize"``,
    and a wrapping shell would leave the real process behind under a different cmdline.
    """
    _check_work(work)
    rendered = " ".join(shlex.quote(a) for a in args)
    return (
        "set -euo pipefail\n"
        f"if [ -f '{work}/env.sh' ]; then . '{work}/env.sh'; fi\n"
        f"exec {work}/venv/bin/infervolt optimize --home '{work}/state'"
        + (f" {rendered}" if rendered else "")
        + "\n"
    )


def kill_command() -> str:
    """Stop the remote run without waiting for SSH to notice we hung up.

    ``|| true`` because the process may already be gone -- the cap firing a second after
    the run finished is not an error worth failing the teardown for.
    """
    return 'pkill -f "infervolt optimize" || true'


__all__ = [
    "BOOTSTRAP_OK",
    "DEFAULT_VLLM_VERSION",
    "REPO_URL",
    "WORK_MARKER",
    "EngineInstall",
    "bootstrap_script",
    "controller_sha",
    "kill_command",
    "optimize_command",
    "passthrough_env",
    "stamp",
]
