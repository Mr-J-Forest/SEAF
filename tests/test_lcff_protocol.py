import json
from pathlib import Path

from config import DEFAULT_CONFIG, load_config, merge_configs
from scripts.run_experiment_queue import expand_matrix
from seaf_model import SEAFNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_SLICES = {
    "TEMP": [0, 20],
    "SALT": [20, 40],
    "UVEL": [40, 60],
    "VVEL": [60, 80],
    "SSHA": [80, 81],
    "MLD": [81, 82],
    "TAUX": [82, 83],
    "TAUY": [83, 84],
    "QNET": [84, 85],
    "WFLUX": [85, 86],
    "TENDENCY_TEMP": [86, 106],
    "TENDENCY_SALT": [106, 126],
}


def resolved_model(config_name: str) -> SEAFNet:
    path = PROJECT_ROOT / "configs" / "experiments" / config_name
    config = merge_configs(load_config(path), DEFAULT_CONFIG.copy())
    config.update({
        "actual_input_channels": 126,
        "actual_output_channels": 40,
        "input_channel_slices": CHANNEL_SLICES,
    })
    return SEAFNet(config)


def test_lcff_off_preserves_external_inputs_in_shared_context() -> None:
    model = resolved_model("oras5_seaf_no_lcff.json")
    encoder = model.spectral_encoder

    assert model.router_type == "spatial"
    assert model.lead_router_logits is None
    assert model.ensemble_gate is not None
    assert encoder.forcing_encoder is None
    assert encoder.forcing_indices.numel() == 0
    assert len(encoder.context_indices) == 86
    assert model.parameter_breakdown()["total"] == 2_205_118


def test_lcff_on_jointly_enables_forcing_branch_and_lead_router() -> None:
    model = resolved_model("oras5_seaf_lcff.json")
    encoder = model.spectral_encoder

    assert model.router_type == "lead"
    assert model.lead_router_logits is not None
    assert tuple(model.lead_router_logits.shape) == (5, 4)
    assert model.ensemble_gate is None
    assert encoder.forcing_encoder is not None
    assert len(encoder.forcing_indices) == 46
    assert len(encoder.context_indices) == 40
    assert model.parameter_breakdown()["total"] == 2_211_351


def test_lcff_confirmation_matrix_contains_only_missing_runs() -> None:
    path = PROJECT_ROOT / "configs" / "oras5_seaf_lcff_confirmation.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    jobs = expand_matrix(matrix, ["confirm_validation"], only=None)

    assert len(jobs) == 4
    assert {(job["name"], job["seed"]) for job in jobs} == {
        ("seaf_no_lcff", 123),
        ("seaf_no_lcff", 3407),
        ("seaf_lcff", 123),
        ("seaf_lcff", 3407),
    }
    assert all(job["overrides"]["epochs"] == 30 for job in jobs)
    assert all(
        job["overrides"]["post_training_evaluation"] == "validation"
        for job in jobs
    )


def test_h192_is_a_width_only_complete_seaf_candidate() -> None:
    model = resolved_model("oras5_seaf_h192.json")

    assert model.router_type == "lead"
    assert model.use_forcing_encoder is True
    assert model.use_temporal_depth_mixer is True
    assert model.use_local_path is True
    assert model.parameter_breakdown()["total"] == 4_972_791


def test_h192_confirmation_matrix_contains_three_validation_runs() -> None:
    path = PROJECT_ROOT / "configs" / "oras5_seaf_h192_confirmation.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    jobs = expand_matrix(matrix, ["confirm_validation"], only=None)

    assert {(job["name"], job["seed"]) for job in jobs} == {
        ("seaf_h192", 42),
        ("seaf_h192", 123),
        ("seaf_h192", 3407),
    }
    assert all(job["overrides"]["epochs"] == 30 for job in jobs)
    assert all(
        job["overrides"]["post_training_evaluation"] == "validation"
        for job in jobs
    )
