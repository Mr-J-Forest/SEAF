import json
import sys

import scripts.generate_dynaseaf_paper_tables as generator


def test_generator_emits_complete_machine_contract_without_science_fillers(
    tmp_path, monkeypatch
):
    output = tmp_path / "dynaseaf_results"
    monkeypatch.setattr(generator, "load_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_dynaseaf_paper_tables.py",
            "--results",
            str(tmp_path / "missing_runs"),
            "--output",
            str(output),
        ],
    )

    assert generator.main() == 0

    expected_csv = {
        "per_seed.csv",
        "lead_skill.csv",
        "depth_skill.csv",
        "lead_depth_skill.csv",
        "dynamics_metrics.csv",
        "gate_statistics.csv",
        "deformation_statistics.csv",
    }
    assert expected_csv.issubset({path.name for path in output.glob("*.csv")})
    for filename in expected_csv:
        assert "TODO_FROM_FROZEN_RESULTS" in (output / filename).read_text(
            encoding="utf-8-sig"
        )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "TODO_FROM_FROZEN_RESULTS"
    assert set(summary["machine_readable_contract"]["csv_files"]) == expected_csv

    bootstrap = json.loads(
        (output / "paired_bootstrap.json").read_text(encoding="utf-8")
    )
    assert bootstrap["status"] == "TODO_FROM_FROZEN_RESULTS"
    assert bootstrap["protocol"]["replicates"] == 10000
    assert json.loads(
        (output / "parameter_count.json").read_text(encoding="utf-8")
    )["status"] == "TODO_FROM_FROZEN_RESULTS"
