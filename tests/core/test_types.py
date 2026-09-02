import typing

import pytest
from pydantic import ValidationError

from infervolt.core.types import (
    BOTTLENECK_PRIORITY,
    SLO,
    Bottleneck,
    EngineConfig,
    Knob,
    KnobSpace,
    RequestRecord,
)


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


def test_request_record_serializes_e2e_and_round_trips() -> None:
    rec = RequestRecord(ttft_s=0.5, itl_s=[0.01, 0.02], output_tokens=2)
    dumped = rec.model_dump()
    assert "e2e_s" in dumped
    assert abs(dumped["e2e_s"] - 0.53) < 1e-9
    assert RequestRecord.model_validate(dumped) == rec


def test_knob_space_get_raises_on_unknown_name() -> None:
    space = _space()
    assert space.get("a").kind == "int"
    with pytest.raises(KeyError):
        space.get("nope")


def test_knob_space_groups_preserves_order_and_dedups() -> None:
    assert _space().groups() == ["kv", "sched", "decode"]


def test_slo_accepts_percentile_and_goodput_target() -> None:
    slo = SLO(ttft_ms=250.0, percentile=0.99, goodput_target=0.95)
    assert slo.percentile == 0.99
    assert slo.goodput_target == 0.95
    assert SLO().goodput_target == 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"percentile": 95.0},
        {"percentile": 0.0},
        {"percentile": -0.1},
        {"goodput_target": 1.5},
        {"goodput_target": 0.0},
    ],
)
def test_slo_fractions_must_be_in_zero_to_one(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SLO(**kwargs)


def test_bottleneck_priority_covers_every_bottleneck() -> None:
    assert set(typing.get_args(Bottleneck)) == set(BOTTLENECK_PRIORITY)


def test_cat_knob_requires_default_in_choices() -> None:
    with pytest.raises(ValidationError):
        Knob(name="c", kind="cat", groups=["kv"], default="nope", choices=["auto", "fp8"])
    with pytest.raises(ValidationError):
        Knob(name="c", kind="cat", groups=["kv"], default="auto")


def test_numeric_knob_requires_bounds_containing_default() -> None:
    with pytest.raises(ValidationError):
        Knob(name="a", kind="int", groups=["kv"], default=8, low=1)
    with pytest.raises(ValidationError):
        Knob(name="a", kind="int", groups=["kv"], default=8, low=16, high=64)
    with pytest.raises(ValidationError):
        Knob(name="a", kind="float", groups=["kv"], default=0.5, low=1.0, high=0.1)


def test_bool_knob_requires_bool_default() -> None:
    with pytest.raises(ValidationError):
        Knob(name="b", kind="bool", groups=["sched"], default="yes")


def test_zero_low_bound_is_accepted() -> None:
    knob = Knob(name="a", kind="int", groups=["kv"], default=0, low=0, high=8)
    assert knob.low == 0
