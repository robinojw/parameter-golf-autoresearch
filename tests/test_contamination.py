# tests/test_contamination.py
import textwrap
from pathlib import Path

from compute.contamination import (
    check_data_overlap,
    check_score_plausibility,
    ContaminationResult,
)


class TestCheckDataOverlap:
    def test_clean_script_passes(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(textwrap.dedent("""\
            import mlx
            train_data = load("data/train.bin")
            for batch in train_data:
                loss = model(batch)
                loss.backward()
        """))
        result = check_data_overlap(script, val_paths=["data/val.bin"])
        assert result.status == "pass"
        assert result.references == []

    def test_val_path_in_training_loop_blocks(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(textwrap.dedent("""\
            import mlx
            train_data = load("data/train.bin")
            val_data = load("data/val.bin")
            for batch in train_data:
                loss = model(batch)
                loss.backward()
            # TTT: adapt on val
            for batch in val_data:
                loss = model(batch)
                loss.backward()
        """))
        result = check_data_overlap(script, val_paths=["data/val.bin"])
        assert result.status == "block"
        assert any("val.bin" in ref for ref in result.references)

    def test_val_in_eval_only_passes(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(textwrap.dedent("""\
            val_data = load("data/val.bin")
            with no_grad():
                val_loss = evaluate(model, val_data)
            print(f"val_loss: {val_loss}")
        """))
        result = check_data_overlap(script, val_paths=["data/val.bin"])
        assert result.status == "pass"

    def test_open_call_with_val_path_blocks(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text(textwrap.dedent("""\
            with open("data/val.bin", "rb") as f:
                val_tokens = f.read()
            # use val_tokens in training
            for step in range(100):
                batch = val_tokens[step*512:(step+1)*512]
                loss = model(batch)
                loss.backward()
        """))
        result = check_data_overlap(script, val_paths=["data/val.bin"])
        assert result.status == "block"


class TestCheckScorePlausibility:
    def test_proportional_improvement_passes(self):
        result = check_score_plausibility(
            train_bpb_before=1.25, train_bpb_after=1.20,
            val_bpb_before=1.22, val_bpb_after=1.18,
        )
        assert result.status == "pass"

    def test_val_improves_much_more_than_train_warns(self):
        result = check_score_plausibility(
            train_bpb_before=1.25, train_bpb_after=1.24,
            val_bpb_before=1.22, val_bpb_after=1.10,
        )
        assert result.status == "warn"

    def test_val_improves_but_train_worsens_blocks(self):
        result = check_score_plausibility(
            train_bpb_before=1.25, train_bpb_after=1.27,
            val_bpb_before=1.22, val_bpb_after=1.10,
        )
        assert result.status == "block"

    def test_no_change_passes(self):
        result = check_score_plausibility(
            train_bpb_before=1.25, train_bpb_after=1.25,
            val_bpb_before=1.22, val_bpb_after=1.22,
        )
        assert result.status == "pass"
