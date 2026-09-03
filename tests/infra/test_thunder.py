"""ThunderProvider against a mock transport. No request in this file leaves the process.

The GET fixtures are real responses captured from the live API; the create and delete
ones are built from the published OpenAPI schemas, because creating an instance costs
money and is never done in a test.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest

from infervolt.config import Settings
from infervolt.infra import thunder as thunder_mod
from infervolt.infra.thunder import ThunderApiError, ThunderProvider, resolve_token
from infervolt.infra.types import InfraError, Instance, InstanceSpec
from infervolt.store.ledger import Ledger

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"thunder_{name}.json").read_text())


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test authenticates from the environment, so none of them can read a real one."""
    monkeypatch.setenv("TNR_API_TOKEN", "test-token")


def make_provider(
    handler: Any, tmp_path: Path, ledger: Ledger | None = None
) -> tuple[ThunderProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record), base_url=thunder_mod.API_BASE)
    settings = Settings(home=tmp_path)
    return ThunderProvider(ledger, settings=settings, client=client), seen


def seed_key(tmp_path: Path, material: str = "ssh-ed25519 AAAAPLACEHOLDER infervolt\n") -> None:
    """Pre-place our public key so a test never shells out to ssh-keygen."""
    (tmp_path / "keys").mkdir(exist_ok=True)
    (tmp_path / "keys" / "id_ed25519.pub").write_text(material)


def catalogue(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/pricing"):
        return httpx.Response(200, json=fixture("pricing"))
    if request.url.path.endswith("/specs"):
        return httpx.Response(200, json=fixture("specs"))
    raise AssertionError(f"unexpected request {request.url.path}")


# ---- auth
def test_token_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNR_API_TOKEN", "from-env")
    monkeypatch.setenv("INFERVOLT_THUNDER_API_TOKEN", "from-settings")
    assert resolve_token(Settings()) == ("from-env", "env TNR_API_TOKEN")


def test_settings_token_is_used_when_the_tnr_variable_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TNR_API_TOKEN", raising=False)
    monkeypatch.setenv("INFERVOLT_THUNDER_API_TOKEN", "from-settings")
    token, source = resolve_token(Settings())
    assert (token, source) == ("from-settings", "env INFERVOLT_THUNDER_API_TOKEN")


def test_the_tnr_credential_file_is_the_last_resort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TNR_API_TOKEN", raising=False)
    config = tmp_path / "cli_config.json"
    config.write_text(json.dumps({"token": "from-file", "expires_at": 1}))
    monkeypatch.setattr(thunder_mod, "TNR_CONFIG", config)
    token, source = resolve_token(Settings())
    assert token == "from-file"
    # Only the *source* may ever be logged, so it must not carry the token.
    assert source == "file"


def test_no_token_anywhere_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TNR_API_TOKEN", raising=False)
    monkeypatch.setattr(thunder_mod, "TNR_CONFIG", tmp_path / "absent.json")
    with pytest.raises(InfraError, match="TNR_API_TOKEN"):
        resolve_token(Settings())


def test_every_request_carries_a_user_agent_and_the_bearer_token(tmp_path: Path) -> None:
    # Thunder answers 403 to a request without a User-Agent, whatever the token is.
    provider, seen = make_provider(catalogue, tmp_path)
    provider.list_offers()
    assert seen and all(r.headers["user-agent"].startswith("infervolt/") for r in seen)
    assert all(r.headers["authorization"] == "Bearer test-token" for r in seen)


def test_the_token_source_is_logged_but_never_the_token(tmp_path: Path) -> None:
    lines: list[str] = []
    provider, _ = make_provider(catalogue, tmp_path)
    provider.log = lines.append
    provider.list_offers()
    assert lines == ["thunder token source: env TNR_API_TOKEN"]
    assert not any("test-token" in line for line in lines)


# ---- offers
def test_offers_are_parsed_from_the_live_pricing_and_specs_shapes(tmp_path: Path) -> None:
    provider, _ = make_provider(catalogue, tmp_path)
    offers = provider.list_offers()
    by_id = {o.raw_id: o for o in offers}

    a100 = by_id["a100xl_x1"]
    assert (a100.gpu, a100.gpu_mem_gb, a100.count) == ("NVIDIA A100 (80GB)", 80.0, 1)
    assert a100.usd_per_hour == 1.09 and a100.provider == "thunder" and a100.spot is False
    assert by_id["h100_x8"].usd_per_hour == 25.6 and by_id["h100_x8"].count == 8

    # Line items and non-virtualised variants are not offers.
    assert not {"disk_gb", "additional_vcpus", "snapshot_gb", "a100xl_native"} & set(by_id)
    # The bare family key duplicates _x1 at the same price and must not double-count.
    assert "h100" not in by_id and "a100xl" not in by_id


def test_offers_can_be_filtered_by_gpu_family(tmp_path: Path) -> None:
    provider, _ = make_provider(catalogue, tmp_path)
    # a100 is Thunder's 40GB part; the 80GB one users ask for by name is a100xl.
    assert {o.raw_id for o in provider.list_offers("a100")} == {
        "a100xl_x1",
        "a100xl_x2",
        "a100xl_x4",
        "a100xl_x8",
    }
    assert {o.count for o in provider.list_offers("h100")} == {1, 2, 4, 8}
    assert provider.list_offers("mi300x") == []


def test_the_catalogue_is_fetched_once_per_provider(tmp_path: Path) -> None:
    provider, seen = make_provider(catalogue, tmp_path)
    provider.list_offers()
    provider.list_offers("h100")
    assert sorted(r.url.path for r in seen) == ["/v1/pricing", "/v1/specs"]


# ---- list and refresh
def test_refresh_reads_status_and_the_ssh_target_off_the_list_response(tmp_path: Path) -> None:
    provider, _ = make_provider(
        lambda r: httpx.Response(200, json=fixture("instances_list")), tmp_path
    )
    inst = Instance(provider="thunder", id="42", gpu="a100xl", raw={"key_path": "/k/thunder-42"})
    fresh = provider.refresh(inst)
    assert fresh.status == "running" and fresh.count == 1
    assert fresh.ssh is not None
    assert (fresh.ssh.host, fresh.ssh.port, fresh.ssh.user) == ("203.0.113.10", 22, "ubuntu")
    assert fresh.ssh.key_path == "/k/thunder-42"


def test_an_instance_without_an_ip_yet_has_no_ssh_target(tmp_path: Path) -> None:
    provider, _ = make_provider(
        lambda r: httpx.Response(200, json=fixture("instances_list")), tmp_path
    )
    inst = Instance(provider="thunder", id="43", gpu="h100", raw={"key_path": "/k/thunder-43"})
    fresh = provider.refresh(inst)
    assert fresh.status == "pending" and fresh.ssh is None


def test_an_instance_the_api_no_longer_lists_reads_as_terminated(tmp_path: Path) -> None:
    # The live response for "you have nothing running" is {}, not a list or an error.
    provider, _ = make_provider(
        lambda r: httpx.Response(200, json=fixture("instances_list_empty")), tmp_path
    )
    inst = Instance(provider="thunder", id="42", gpu="a100xl", status="running")
    assert provider.refresh(inst).status == "terminated"


# ---- provision
def _create_handler(created: list[dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/instances/create"):
            created.append(json.loads(request.content))
            return httpx.Response(201, json=fixture("create"))
        return catalogue(request)

    return handler


def test_provision_sends_the_documented_body_and_records_the_instance(tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []
    seed_key(tmp_path)
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider, _ = make_provider(_create_handler(created), tmp_path, ledger)
        inst = provider.provision(InstanceSpec(gpu="a100-80", count=1, disk_gb=250))

        body = created[0]
        assert body["gpu_type"] == "a100xl" and body["num_gpus"] == 1
        assert body["template"] == "ubuntu-22.04" and body["mode"] == "production"
        assert body["disk_size_gb"] == 250
        # cpu_cores comes from the production spec for a100xl_x1, which offers only 15.
        assert body["cpu_cores"] == 15
        assert body["public_key"].startswith("ssh-")

        assert inst.id == "42" and inst.gpu == "a100xl" and inst.usd_per_hour == 1.09
        assert inst.status == "provisioning" and inst.raw["offer_id"] == "a100xl_x1"
        assert [(i.provider, i.id) for i in ledger.instances()] == [("thunder", "42")]


def test_a_returned_private_key_is_stored_readable_only_by_us(tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []
    seed_key(tmp_path)
    provider, _ = make_provider(_create_handler(created), tmp_path)
    inst = provider.provision(InstanceSpec(gpu="a100-80"))
    key = Path(inst.raw["key_path"])
    assert key == tmp_path / "keys" / "thunder-42"
    assert key.read_text() == fixture("create")["key"]
    assert stat.S_IMODE(key.stat().st_mode) == 0o600


def test_disk_is_clamped_to_what_the_configuration_allows(tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []
    seed_key(tmp_path)
    provider, _ = make_provider(_create_handler(created), tmp_path)
    provider.provision(InstanceSpec(gpu="a100-80", disk_gb=9000))
    assert created[0]["disk_size_gb"] == 500  # storageGB.max for a100xl_x1
    provider.provision(InstanceSpec(gpu="a100-80", disk_gb=10))
    assert created[1]["disk_size_gb"] == 100  # storageGB.min


def test_our_public_key_is_generated_once_and_then_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, Any]] = []
    provider, _ = make_provider(_create_handler(created), tmp_path)
    seed_key(tmp_path)

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("ssh-keygen must not run when a key already exists")

    monkeypatch.setattr(thunder_mod.subprocess, "run", explode)
    provider.provision(InstanceSpec(gpu="a100-80"))
    assert created[0]["public_key"] == "ssh-ed25519 AAAAPLACEHOLDER infervolt"


def test_a_missing_key_is_generated_into_our_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key that opens rented boxes is ours, not the user's ~/.ssh identity."""
    created: list[dict[str, Any]] = []
    provider, _ = make_provider(_create_handler(created), tmp_path)
    argv: list[list[str]] = []

    def fake_keygen(cmd: list[str], **kwargs: object) -> Any:
        argv.append(cmd)
        target = Path(cmd[cmd.index("-f") + 1])
        target.write_text("private")
        target.with_name(target.name + ".pub").write_text("ssh-ed25519 GENERATED infervolt\n")
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(thunder_mod.subprocess, "run", fake_keygen)
    provider.provision(InstanceSpec(gpu="a100-80"))
    assert argv[0][:2] == ["ssh-keygen", "-t"]
    assert argv[0][argv[0].index("-f") + 1] == str(tmp_path / "keys" / "id_ed25519")
    assert argv[0][argv[0].index("-N") + 1] == ""  # no passphrase: nothing can type one in
    assert created[0]["public_key"] == "ssh-ed25519 GENERATED infervolt"
    assert stat.S_IMODE((tmp_path / "keys" / "id_ed25519").stat().st_mode) == 0o600


def test_a_failing_keygen_is_a_provision_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        thunder_mod.subprocess,
        "run",
        lambda *a, **k: type("Completed", (), {"returncode": 1, "stderr": "no space"})(),
    )
    provider, _ = make_provider(_create_handler([]), tmp_path)
    with pytest.raises(thunder_mod.ProvisionError, match="ssh-keygen"):
        provider.provision(InstanceSpec(gpu="a100-80"))


def test_a_caller_supplied_public_key_wins(tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []
    provider, _ = make_provider(_create_handler(created), tmp_path)
    provider.provision(InstanceSpec(gpu="h100", ssh_public_key="ssh-ed25519 CALLER x"))
    assert created[0]["public_key"] == "ssh-ed25519 CALLER x"


def test_a_create_response_without_an_identifier_is_a_provision_error(tmp_path: Path) -> None:
    provider, _ = make_provider(
        lambda r: (
            httpx.Response(201, json={"uuid": "u"})
            if r.url.path.endswith("create")
            else catalogue(r)
        ),
        tmp_path,
    )
    with pytest.raises(thunder_mod.ProvisionError, match="identifier"):
        provider.provision(InstanceSpec(gpu="a100-80", ssh_public_key="ssh-ed25519 X y"))


# ---- terminate
def test_terminate_calls_delete_and_closes_the_ledger_row(tmp_path: Path) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"message": "Success"})

    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider, _ = make_provider(handler, tmp_path, ledger)
        inst = Instance(provider="thunder", id="42", gpu="a100xl", usd_per_hour=1.09)
        ledger.save_instance(inst)
        provider.terminate(inst)
        assert paths == ["/v1/instances/42/delete"]
        assert ledger.instances() == []
        assert [i.id for i in ledger.instances(active_only=False)] == ["42"]


def test_terminating_something_already_gone_is_not_an_error(tmp_path: Path) -> None:
    with Ledger(tmp_path / "l.sqlite", tmp_path / "runs") as ledger:
        provider, _ = make_provider(
            lambda r: httpx.Response(404, json={"message": "not found"}), tmp_path, ledger
        )
        inst = Instance(provider="thunder", id="42", gpu="a100xl")
        ledger.save_instance(inst)
        provider.terminate(inst)
        assert ledger.instances() == []


def test_a_real_api_failure_on_delete_is_raised(tmp_path: Path) -> None:
    provider, _ = make_provider(lambda r: httpx.Response(500, json={"message": "boom"}), tmp_path)
    with pytest.raises(ThunderApiError, match="boom") as excinfo:
        provider.terminate(Instance(provider="thunder", id="42", gpu="a100xl"))
    assert excinfo.value.status == 500


def test_an_unauthorised_response_says_so(tmp_path: Path) -> None:
    provider, _ = make_provider(lambda r: httpx.Response(403, text="Forbidden"), tmp_path)
    with pytest.raises(ThunderApiError, match="403"):
        provider.list_offers()


# ---- pricing
def test_cost_per_hour_falls_back_to_the_price_list(tmp_path: Path) -> None:
    provider, _ = make_provider(catalogue, tmp_path)
    priced = Instance(provider="thunder", id="1", gpu="a100xl", usd_per_hour=2.5)
    assert provider.cost_per_hour(priced) == 2.5
    unpriced = Instance(provider="thunder", id="2", gpu="a100xl", count=2)
    assert provider.cost_per_hour(unpriced) == 2.18
