import json
import sys

import scripts.validate_experiment_matrix as validator


def test_dynaseaf_formal_stages_resolve_to_30_epochs():
    for filename, stages in (
        (
            "configs/oras5_dynaseaf_screen_matrix.json",
            ("screen",),
        ),
        (
            "configs/oras5_dynaseaf_remaining_screen_matrix.json",
            ("screen",),
        ),
        (
            "configs/oras5_dynaseaf_ablation_matrix.json",
            ("screen", "confirm_validation"),
        ),
    ):
        matrix = json.loads(open(filename, encoding="utf-8").read())
        for stage in stages:
            jobs = validator.expand_matrix(matrix, [stage], only=None)
            assert jobs
            assert all(job["overrides"]["epochs"] == 30 for job in jobs)


def test_validator_rejects_an_80_epoch_formal_stage(tmp_path, monkeypatch, capsys):
    matrix_path = tmp_path / "stale_matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "_contrasts": None,
                "_protocol": {"formal_epochs": 30},
                "_stage_overrides": {
                    "confirm_validation": {
                        "epochs": 80,
                        "post_training_evaluation": "validation",
                    }
                },
                "confirm_validation": [
                    {
                        "name": "full",
                        "config": "configs/experiments/oras5_seaf_h192.json",
                        "seeds": [42, 123, 3407],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_experiment_matrix.py", "--matrix", str(matrix_path)],
    )

    assert validator.main() == 1
    assert "formal training budget must be 30 epochs" in capsys.readouterr().out
