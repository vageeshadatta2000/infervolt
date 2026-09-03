from infervolt.core.types import Knob, KnobSpace
from infervolt.search.space import Bounds, clamp, is_novel

SPACE = KnobSpace(
    knobs=[
        Knob(
            name="gpu_memory_utilization",
            kind="float",
            groups=["kv"],
            default=0.9,
            low=0.7,
            high=0.95,
            step=0.05,
        ),
        Knob(
            name="max_model_len",
            kind="cat",
            groups=["kv"],
            default=32768,
            choices=[4096, 8192, 16384, 32768],
        ),
        Knob(
            name="max_num_seqs",
            kind="int",
            groups=["kv"],
            default=256,
            low=8,
            high=1024,
            log=True,
        ),
        Knob(
            name="kv_cache_dtype",
            kind="cat",
            groups=["kv"],
            default="auto",
            choices=["auto", "fp8"],
        ),
    ]
)


def test_tighten_on_oom_lowers_upper_bounds_and_clamps() -> None:
    b = Bounds(SPACE)
    b.tighten_on_oom(
        {
            "gpu_memory_utilization": 0.95,
            "max_model_len": 32768,
            "max_num_seqs": 512,
            "kv_cache_dtype": "auto",
        }
    )
    assert b.high["gpu_memory_utilization"] == 0.9
    assert b.high["max_model_len"] == 16384
    assert b.high["max_num_seqs"] == 511
    out = clamp(
        {
            "gpu_memory_utilization": 0.95,
            "max_model_len": 32768,
            "max_num_seqs": 900,
            "kv_cache_dtype": "fp8",
        },
        SPACE,
        b,
    )
    assert out == {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16384,
        "max_num_seqs": 511,
        "kv_cache_dtype": "fp8",
    }


def test_tighten_on_oom_never_raises_the_ceiling() -> None:
    """A later, smaller OOM tightens further; a larger one leaves the ceiling alone."""
    b = Bounds(SPACE)
    b.tighten_on_oom({"max_num_seqs": 64})
    b.tighten_on_oom({"max_num_seqs": 900})
    assert b.high["max_num_seqs"] == 63


def test_tighten_never_pushes_a_ceiling_below_the_knobs_floor() -> None:
    """An OOM blames every memory knob at once, so it must not empty one knob's range."""
    b = Bounds(SPACE)
    b.tighten_on_oom(
        {"gpu_memory_utilization": 0.7, "max_model_len": 4096, "max_num_seqs": 8},
    )
    assert b.high == {
        "gpu_memory_utilization": 0.95,
        "max_model_len": 32768,
        "max_num_seqs": 1024,
    }
    knobs = {
        "gpu_memory_utilization": 0.95,
        "max_model_len": 32768,
        "max_num_seqs": 1024,
        "kv_cache_dtype": "auto",
    }
    assert clamp(knobs, SPACE, b) == knobs


def test_bounds_ignore_non_numeric_knobs() -> None:
    b = Bounds(SPACE)
    assert "kv_cache_dtype" not in b.high
    b.tighten_on_oom({"kv_cache_dtype": "fp8"})
    assert "kv_cache_dtype" not in b.high


def test_clamp_leaves_a_config_inside_the_bounds_untouched() -> None:
    knobs = {
        "gpu_memory_utilization": 0.8,
        "max_model_len": 8192,
        "max_num_seqs": 32,
        "kv_cache_dtype": "auto",
    }
    assert clamp(knobs, SPACE, Bounds(SPACE)) == knobs


def test_novelty_uses_normalized_distance() -> None:
    seen = [
        {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
            "max_num_seqs": 256,
            "kv_cache_dtype": "auto",
        }
    ]
    near = {**seen[0], "gpu_memory_utilization": 0.905}
    far = {**seen[0], "kv_cache_dtype": "fp8"}
    assert not is_novel(near, seen, SPACE)
    assert is_novel(far, seen, SPACE)


def test_novelty_against_an_empty_history_is_always_true() -> None:
    assert is_novel({"gpu_memory_utilization": 0.9}, [], SPACE)


def test_novelty_uses_the_log_scale_for_log_knobs() -> None:
    """8 -> 16 is one octave of seven: far apart on a log knob, adjacent on a linear one."""
    seen = [{"max_num_seqs": 8}]
    assert is_novel({"max_num_seqs": 16}, seen, SPACE)
    assert not is_novel({"max_num_seqs": 9}, seen, SPACE)


WEIGHTS_LOG = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.1 GiB"
KV_LOG = (
    "ValueError: The model's max seq len (32768) is larger than the maximum number of "
    "tokens that can be stored in KV cache (9000)."
)


def test_weights_oom_raises_the_utilization_floor_and_spares_the_token_ceilings() -> None:
    """Too little budget for weights means *more* util, not less -- and the tokens are innocent."""
    b = Bounds(SPACE)
    b.tighten_for(
        WEIGHTS_LOG,
        {"gpu_memory_utilization": 0.7, "max_model_len": 32768, "max_num_seqs": 512},
    )
    assert b.low["gpu_memory_utilization"] == 0.75
    assert b.high == {
        "gpu_memory_utilization": 0.95,
        "max_model_len": 32768,
        "max_num_seqs": 1024,
    }
    # The floor is honoured on the way out, so the sampler cannot revisit the dead region.
    out = clamp({"gpu_memory_utilization": 0.7, "max_model_len": 4096, "max_num_seqs": 8}, SPACE, b)
    assert out["gpu_memory_utilization"] == 0.75
    assert out["max_model_len"] == 4096 and out["max_num_seqs"] == 8


def test_weights_oom_floor_only_rises_and_never_passes_the_ceiling() -> None:
    b = Bounds(SPACE)
    b.tighten_for(WEIGHTS_LOG, {"gpu_memory_utilization": 0.85})
    b.tighten_for(WEIGHTS_LOG, {"gpu_memory_utilization": 0.7})  # a lesser lesson
    assert b.low["gpu_memory_utilization"] == 0.9
    b.tighten_for(WEIGHTS_LOG, {"gpu_memory_utilization": 0.95})
    assert b.low["gpu_memory_utilization"] == 0.95  # capped at the knob's own ceiling


def test_kv_oom_tightens_the_token_ceilings_and_spares_the_utilization() -> None:
    """The KV cache could not hold the context: shorten it, and leave the budget alone."""
    b = Bounds(SPACE)
    b.tighten_for(
        KV_LOG, {"gpu_memory_utilization": 0.9, "max_model_len": 32768, "max_num_seqs": 512}
    )
    assert b.high["max_model_len"] == 16384 and b.high["max_num_seqs"] == 511
    assert b.high["gpu_memory_utilization"] == 0.95
    assert b.low["gpu_memory_utilization"] == 0.7


def test_an_unrecognised_oom_falls_back_to_the_conservative_rule() -> None:
    b = Bounds(SPACE)
    b.tighten_for(
        "Segmentation fault (core dumped)",
        {"gpu_memory_utilization": 0.95, "max_model_len": 32768, "max_num_seqs": 512},
    )
    assert b.high["gpu_memory_utilization"] == 0.9
    assert b.high["max_model_len"] == 16384
    assert b.high["max_num_seqs"] == 511
