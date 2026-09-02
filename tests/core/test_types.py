from infervolt.core.types import EngineConfig, Knob, KnobSpace, RequestRecord


def _space() -> KnobSpace:
    return KnobSpace(
        knobs=[
            Knob(name="a", kind="int", groups=["kv"], default=8, low=1, high=64, log=True),
            Knob(name="b", kind="bool", groups=["sched"], default=False),
            Knob(
                name="c",
                kind="cat",
                groups=["kv", "decode"],
                default="auto",
                choices=["auto", "fp8"],
            ),
        ]
    )


def test_subspace_filters_by_group_membership() -> None:
    sub = _space().subspace(["decode"])
    assert sub.names() == ["c"]
    assert _space().subspace(["kv"]).names() == ["a", "c"]


def test_defaults_and_with_knobs() -> None:
    cfg = EngineConfig(engine="mock", knobs=_space().defaults())
    assert cfg.knobs == {"a": 8, "b": False, "c": "auto"}
    new = cfg.with_knobs(c="fp8")
    assert new.knobs["c"] == "fp8" and cfg.knobs["c"] == "auto"


def test_config_key_is_order_independent() -> None:
    k1 = EngineConfig(engine="mock", knobs={"x": 1, "y": 2}).key()
    k2 = EngineConfig(engine="mock", knobs={"y": 2, "x": 1}).key()
    assert k1 == k2


def test_request_record_e2e() -> None:
    r = RequestRecord(ttft_s=0.5, itl_s=[0.01, 0.02], output_tokens=2)
    assert abs(r.e2e_s - 0.53) < 1e-9
