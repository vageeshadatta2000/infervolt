"""Thunder Compute, over its REST API.

REST rather than the ``tnr`` CLI: the CLI needs a sudo reinstall to reach a version that
speaks JSON everywhere, and the API is what the CLI calls anyway. The field names below
were read off a live ``GET /pricing``, ``GET /instances/list`` and ``GET /specs`` plus the
published OpenAPI document, not guessed; the captured responses are in
``tests/infra/fixtures``.

Two Thunder-specific facts are encoded here and nowhere else:

* a request with no ``User-Agent`` is answered 403 no matter how good the token is;
* SSH keys stored in the account are *not* attached to new instances -- ``public_key``
  must be sent on every create, or Thunder mints a keypair and returns the private half
  exactly once.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from infervolt import __version__
from infervolt.config import Settings, get_settings
from infervolt.infra.base import Provider
from infervolt.infra.types import (
    InfraError,
    Instance,
    InstanceSpec,
    LaunchMode,
    Offer,
    ProvisionError,
    SshTarget,
)

if TYPE_CHECKING:  # pragma: no cover
    from infervolt.store.ledger import Ledger

API_BASE = "https://api.thundercompute.com:8443/v1"
USER_AGENT = f"infervolt/{__version__}"
TNR_CONFIG = Path.home() / ".thunder" / "cli_config.json"
SSH_USER = "ubuntu"
DEFAULT_TEMPLATE = "ubuntu-22.04"
DEFAULT_MODE = "production"

GPU_ALIASES = {
    "t4": "t4",
    "a100": "a100",
    "a100-40": "a100",
    "a100xl": "a100xl",
    "a100-80": "a100xl",
    "h100": "h100",
    "h100-80": "h100",
    "l40": "l40",
    "l40s": "l40",
    "a6000": "a6000",
}
"""How a user's GPU name becomes a Thunder ``gpu_type``.

``a100xl`` is Thunder's name for the 80GB A100 and ``a100`` for the 40GB one, which is
the one mapping nobody guesses right.
"""

GPU_FALLBACK = {
    "t4": ("NVIDIA T4", 16.0),
    "a100": ("NVIDIA A100 (40GB)", 40.0),
    "a100xl": ("NVIDIA A100 (80GB)", 80.0),
    "h100": ("NVIDIA H100", 80.0),
    "l40": ("NVIDIA L40S", 48.0),
    "a6000": ("RTX A6000", 48.0),
}
"""Display name and VRAM for families ``/specs`` does not describe.

``/specs`` is the authority wherever it has an entry; this table only stops a priced
family from disappearing out of ``infra offers`` because the specs endpoint lags.
"""


class ThunderApiError(InfraError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def split_pricing_key(key: str) -> tuple[str, int] | None:
    """``a100xl_x2`` -> ``("a100xl", 2)``; ``disk_gb`` -> ``None``.

    The pricing map mixes GPU configurations with line items (``additional_vcpus``,
    ``disk_gb``, ``snapshot_gb``) and with non-virtualised variants (``a100xl_native``)
    that cannot be ordered through ``num_gpus``. Only ``<family>`` and ``<family>_x<N>``
    are offers.
    """
    family, sep, suffix = key.rpartition("_x")
    if sep and suffix.isdigit():
        return family, int(suffix)
    if "_" in key:
        return None
    return key, 1


def resolve_token(settings: Settings | None = None) -> tuple[str, str]:
    """Find a Thunder token; return it with the *name* of where it came from.

    The name, never the value, is what gets logged. The ambient ``TNR_API_TOKEN`` wins so
    that a shell already set up for the ``tnr`` CLI needs no second configuration step,
    and the credential file is last because reading another tool's secrets is the step a
    user should be able to override without deleting anything.
    """
    env = os.environ.get("TNR_API_TOKEN")
    if env:
        return env, "env TNR_API_TOKEN"
    settings = settings or get_settings()
    if settings.thunder_api_token is not None:
        return settings.thunder_api_token.get_secret_value(), "env INFERVOLT_THUNDER_API_TOKEN"
    if TNR_CONFIG.is_file():
        try:
            payload = json.loads(TNR_CONFIG.read_text())
        except (OSError, ValueError) as e:
            raise InfraError(f"could not read the tnr credential file: {e}") from None
        token = payload.get("token") if isinstance(payload, dict) else None
        if token:
            return str(token), "file"
    raise InfraError(
        "no Thunder token: set TNR_API_TOKEN or INFERVOLT_THUNDER_API_TOKEN, or log in with tnr"
    )


class ThunderProvider(Provider):
    """Virtualised NVIDIA GPUs, billed per minute, reached over plain SSH as ``ubuntu``."""

    name: ClassVar[str] = "thunder"
    launch_mode: ClassVar[LaunchMode] = "vm"
    supports_scp: ClassVar[bool] = True

    def __init__(
        self,
        ledger: Ledger | None = None,
        *,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(ledger)
        self.settings = settings or get_settings()
        self.log = log
        self._client = client
        self._token: str | None = None
        self._specs: dict[str, dict[str, Any]] | None = None
        self._pricing: dict[str, float] | None = None

    # ---- HTTP
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=API_BASE, timeout=30.0)
        return self._client

    def _token_value(self) -> str:
        if self._token is None:
            token, source = resolve_token(self.settings)
            if self.log is not None:
                self.log(f"thunder token source: {source}")
            self._token = token
        return self._token

    def _headers(self) -> dict[str, str]:
        # User-Agent is not politeness here: Thunder answers 403 without one.
        return {
            "Authorization": f"Bearer {self._token_value()}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, headers=self._headers(), json=body)
        except httpx.HTTPError as e:
            raise InfraError(f"thunder {method} {path} failed: {e}") from None
        if response.status_code >= 400:
            raise ThunderApiError(
                response.status_code,
                f"thunder {method} {path} -> {response.status_code}: {_why(response)}",
            )
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError:
            raise InfraError(f"thunder {method} {path} returned non-JSON") from None
        if not isinstance(payload, dict):
            raise InfraError(f"thunder {method} {path} returned {type(payload).__name__}, not JSON")
        result: dict[str, Any] = payload
        return result

    # ---- catalogue
    def specs(self) -> dict[str, dict[str, Any]]:
        """``GET /specs``, cached: vCPU options, storage bounds and VRAM per offer key."""
        if self._specs is None:
            raw = self._request("GET", "/specs").get("specs", {})
            entries: dict[str, dict[str, Any]] = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(value, dict):
                        entries[str(key)] = value
            self._specs = entries
        return self._specs

    def pricing(self) -> dict[str, float]:
        """``GET /pricing``, cached: USD per hour per offer key, plus storage line items."""
        if self._pricing is None:
            raw = self._request("GET", "/pricing").get("pricing", {})
            prices: dict[str, float] = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(value, int | float):
                        prices[str(key)] = float(value)
            self._pricing = prices
        return self._pricing

    def list_offers(self, gpu: str | None = None) -> list[Offer]:
        pricing = self.pricing()
        specs = self.specs()
        offers: list[Offer] = []
        for key, price in pricing.items():
            split = split_pricing_key(key)
            if split is None:
                continue
            family, count = split
            spec = specs.get(f"{family}_x{count}") or specs.get(key)
            if spec is None and family not in GPU_FALLBACK:
                continue
            # The bare family key duplicates ``<family>_x1`` at the same price; keep the
            # explicit one, so every offer says how many GPUs it is.
            if "_x" not in key and f"{family}_x1" in pricing:
                continue
            name, mem_gb = GPU_FALLBACK.get(family, (family.upper(), 0.0))
            spec = spec or {}
            offers.append(
                Offer(
                    provider=self.name,
                    gpu=str(spec.get("displayName") or name),
                    gpu_mem_gb=float(spec.get("vramGB") or mem_gb),
                    count=int(spec.get("gpuCount") or count),
                    usd_per_hour=price,
                    spot=False,
                    raw_id=key,
                )
            )
        if gpu:
            wanted = GPU_ALIASES.get(gpu.lower(), gpu.lower())
            offers = [o for o in offers if wanted in o.raw_id.lower() or wanted in o.gpu.lower()]
        return sorted(offers, key=lambda o: (o.gpu, o.count))

    # ---- lifecycle
    def provision(self, spec: InstanceSpec) -> Instance:
        family = GPU_ALIASES.get(spec.gpu.lower(), spec.gpu.lower())
        offer_id = f"{family}_x{spec.count}"
        config = self.specs().get(f"{offer_id}_{DEFAULT_MODE}") or self.specs().get(offer_id) or {}
        body = {
            "gpu_type": family,
            "num_gpus": spec.count,
            "cpu_cores": _cpu_cores(config),
            "disk_size_gb": _disk_gb(config, spec.disk_gb),
            "template": spec.image or DEFAULT_TEMPLATE,
            "mode": DEFAULT_MODE,
            "public_key": spec.ssh_public_key or self._public_key(),
        }
        payload = self._request("POST", "/instances/create", body)
        if "identifier" not in payload:
            raise ProvisionError(f"thunder create returned no identifier: {sorted(payload)}")
        inst_id = str(payload["identifier"])
        key_path = self._store_key(inst_id, payload.get("key")) or self._private_key_path()
        return self._record(
            Instance(
                provider=self.name,
                id=inst_id,
                gpu=family,
                count=spec.count,
                usd_per_hour=self.price_of(family, spec.count),
                launch_mode=self.launch_mode,
                status="provisioning",
                raw={
                    "uuid": str(payload.get("uuid", "")),
                    "gpu_type": family,
                    "key_path": str(key_path),
                    "offer_id": offer_id,
                },
            )
        )

    def refresh(self, inst: Instance) -> Instance:
        """Re-read ``/instances/list``; an instance that has vanished reads as terminated.

        Thunder answers with an object keyed by instance id (``{}`` when there are none),
        and the item carries no id of its own -- the key *is* the id.
        """
        raw = self._request("GET", "/instances/list").get(inst.id)
        if not isinstance(raw, dict):
            return inst.model_copy(update={"status": "terminated", "ssh": None})
        ip = str(raw.get("ip") or "")
        key_path = str(inst.raw.get("key_path") or "")
        ssh = (
            SshTarget(host=ip, port=int(raw.get("port") or 22), user=SSH_USER, key_path=key_path)
            if ip and key_path
            else None
        )
        merged = dict(inst.raw)
        merged["listed"] = raw
        return inst.model_copy(
            update={
                "status": str(raw.get("status") or "unknown").lower(),
                "ssh": ssh,
                "count": int(raw.get("numGpus") or inst.count),
                "raw": merged,
            }
        )

    def terminate(self, inst: Instance) -> None:
        try:
            self._request("POST", f"/instances/{inst.id}/delete")
        except ThunderApiError as e:
            # A box that is already gone is the state we were asking for; nothing else is.
            if e.status != 404:
                raise
        self._record_terminated(inst)

    def cost_per_hour(self, inst: Instance) -> float:
        if inst.usd_per_hour:
            return inst.usd_per_hour
        return self.price_of(str(inst.raw.get("gpu_type") or inst.gpu), inst.count)

    def price_of(self, family: str, count: int) -> float:
        pricing = self.pricing()
        for key in (f"{family}_x{count}", family):
            if key in pricing:
                return pricing[key]
        return 0.0

    # ---- keys
    def _private_key_path(self) -> Path:
        return self.settings.keys_dir / "id_ed25519"

    def _public_key(self) -> str:
        """Our own public key, generated on first use.

        Generated rather than borrowed from ``~/.ssh``: the key that opens rented boxes
        should not be the key that opens everything else the user owns.
        """
        private = self._private_key_path()
        public = private.with_name(private.name + ".pub")
        if not public.is_file():
            private.parent.mkdir(parents=True, exist_ok=True)
            private.unlink(missing_ok=True)
            result = subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "infervolt", "-f", str(private)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not public.is_file():
                raise ProvisionError(f"ssh-keygen failed: {result.stderr.strip()[-300:]}")
            private.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return public.read_text().strip()

    def _store_key(self, inst_id: str, material: object) -> Path | None:
        """Persist the private key Thunder mints when we send no ``public_key``.

        It is returned exactly once, so it is written before anything else can fail, and
        with mode 0600 -- ssh refuses a key file that anyone else can read.
        """
        if not isinstance(material, str) or not material.strip():
            return None
        path = self.settings.keys_dir / f"thunder-{inst_id}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(material if material.endswith("\n") else material + "\n")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return path


def _why(response: httpx.Response) -> str:
    """The API's own error message when it sent one, else a short body excerpt."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        message = body.get("message") or body.get("error")
        if message:
            return str(message)
    return str(body)[:200]


def _cpu_cores(config: dict[str, Any], default: int = 8) -> int:
    """The largest vCPU option the chosen configuration allows.

    Largest, not smallest: the CPU side of an inference benchmark -- tokenizer, load
    generator, model download -- is exactly where a stingy vCPU count shows up as a
    latency number the run then blames on the GPU.
    """
    options = config.get("vcpuOptions")
    if isinstance(options, list) and options:
        return max(int(o) for o in options)
    return default


def _disk_gb(config: dict[str, Any], wanted: int) -> int:
    storage = config.get("storageGB")
    if not isinstance(storage, dict):
        return wanted
    return max(int(storage.get("min", wanted)), min(int(storage.get("max", wanted)), wanted))
