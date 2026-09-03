from infervolt.infra.types import Instance, InstanceSpec, Offer, RunResult, SshTarget


def test_instance_round_trips_through_json() -> None:
    inst = Instance(
        provider="thunder",
        id="42",
        gpu="a100xl",
        count=2,
        usd_per_hour=2.18,
        ssh=SshTarget(host="203.0.113.10", port=22, user="ubuntu", key_path="/k/id_ed25519"),
        launch_mode="vm",
        status="running",
        raw={"uuid": "u-1", "offer_id": "a100xl_x2"},
    )
    back = Instance.model_validate_json(inst.model_dump_json())
    assert back == inst
    assert back.ssh is not None and back.ssh.user == "ubuntu"


def test_instance_defaults_are_the_unknown_ones() -> None:
    inst = Instance(provider="local", id="local", gpu="Apple M3")
    assert inst.ssh is None and inst.status == "unknown" and inst.raw == {}
    assert inst.launch_mode == "vm" and inst.count == 1 and inst.usd_per_hour == 0.0
    assert inst.created_at > 0


def test_instance_spec_defaults_open_ssh_and_the_engine_port() -> None:
    spec = InstanceSpec(gpu="a100")
    assert spec.ports == [22, 8000] and spec.disk_gb == 100 and spec.name == "infervolt"
    # Mutable defaults must not be shared between specs.
    spec.ports.append(9000)
    assert InstanceSpec(gpu="a100").ports == [22, 8000]


def test_offer_and_run_result_round_trip() -> None:
    offer = Offer(
        provider="thunder",
        gpu="NVIDIA A100 (80GB)",
        gpu_mem_gb=80,
        usd_per_hour=1.09,
        raw_id="a100xl_x1",
    )
    assert Offer.model_validate_json(offer.model_dump_json()) == offer
    assert offer.region is None and offer.spot is False and offer.count == 1
    result = RunResult(code=0)
    assert RunResult.model_validate_json(result.model_dump_json()) == result
