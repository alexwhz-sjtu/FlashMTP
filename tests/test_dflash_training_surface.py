from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_independent_training_and_benchmark_entrypoints_exist():
    for algorithm in ("dflash", "dflash2", "dspark"):
        trainer = ROOT / "scripts" / "dflash_family" / f"train_{algorithm}.py"
        launcher = (
            ROOT / "scripts" / "dflash_family" / f"run_training_{algorithm}.sh"
        )
        benchmark = ROOT / "evaluation" / f"run_benchmark_{algorithm}.sh"
        assert trainer.is_file()
        assert launcher.is_file()
        assert benchmark.is_file()
        assert f'run("{algorithm}")' in trainer.read_text(encoding="utf-8")


def test_family_training_excludes_removed_objectives():
    source = (
        ROOT / "scripts" / "dflash_family" / "dflash_family_training.py"
    ).read_text(encoding="utf-8")
    assert "dpace" not in source.lower()
    assert "lk_loss" not in source.lower()
