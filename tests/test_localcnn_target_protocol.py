import json
from pathlib import Path

from config import DEFAULT_CONFIG, load_config, merge_configs
from scripts.run_experiment_queue import expand_matrix
from scripts.summarize_localcnn_target_contrast import (
    build_validation_only_archive,
    config_differences,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolved(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / "experiments" / name
    return merge_configs(load_config(path), DEFAULT_CONFIG.copy())


def test_localcnn_full_field_changes_only_target_formulation() -> None:
    anomaly = resolved("oras5_baseline_local_cnn_anomaly.json")
    full_field = resolved("oras5_baseline_local_cnn_full_field.json")
    allowed = {
        "enable_target_climatology_anomaly",
        "ablation_direct_full_field",
    }
    differences = {
        key for key in set(anomaly) | set(full_field)
        if anomaly.get(key) != full_field.get(key)
    }

    assert differences == allowed
    assert anomaly["enable_target_climatology_anomaly"] is True
    assert anomaly["ablation_direct_full_field"] is False
    assert full_field["enable_target_climatology_anomaly"] is False
    assert full_field["ablation_direct_full_field"] is True


def test_localcnn_full_field_matrix_is_validation_only_three_seed() -> None:
    path = PROJECT_ROOT / "configs" / "oras5_local_cnn_full_field_confirmation.json"
    matrix = json.loads(path.read_text(encoding="utf-8"))
    jobs = expand_matrix(matrix, ["confirm_validation"], only=None)

    assert {(job["name"], job["seed"]) for job in jobs} == {
        ("local_cnn_full_field", 42),
        ("local_cnn_full_field", 123),
        ("local_cnn_full_field", 3407),
    }
    assert all(job["overrides"]["epochs"] == 30 for job in jobs)
    assert all(
        job["overrides"]["post_training_evaluation"] == "validation"
        for job in jobs
    )


def test_legacy_missing_defaults_do_not_create_false_scientific_differences() -> None:
    anomaly = resolved("oras5_baseline_local_cnn_anomaly.json")
    full_field = resolved("oras5_baseline_local_cnn_full_field.json")
    legacy_keys = {
        "model_display_name",
        "router_type",
        "seaf_forcing_scale_init",
        "seaf_spectral_scale_init",
        "use_forcing_encoder",
        "use_local_path",
        "use_spectral_path",
        "use_temporal_depth_mixer",
    }
    for key in legacy_keys:
        anomaly.pop(key)

    assert set(config_differences(anomaly, full_field)) == {
        "enable_target_climatology_anomaly",
        "ablation_direct_full_field",
    }


def test_validation_archive_excludes_historical_test_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "derived"
    for seed in (42, 123, 3407):
        run = source / "confirm_validation" / "local_cnn" / f"seed_{seed}"
        run.mkdir(parents=True)
        for name in ("config.json", "run_summary.json", "validation_results.json"):
            (run / name).write_text("{}\n", encoding="utf-8")
        (run / "evaluation_results.json").write_text(
            '{"must_not_be_read": true}\n', encoding="utf-8"
        )

    manifest = build_validation_only_archive(source, "local_cnn", destination)

    assert len(manifest["copied_validation_inputs"]) == 9
    assert len(manifest["excluded_source_artifacts"]) == 3
    assert all(
        item["policy"] == "inventoried_by_path_only; not opened or copied"
        for item in manifest["excluded_source_artifacts"]
    )
    assert not list(destination.rglob("evaluation_results.json"))
