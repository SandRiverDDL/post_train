from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本：{script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareScriptsTest(unittest.TestCase):
    def test_prepare_stage1_data_only_writes_train_and_dev(self) -> None:
        module = load_script_module("prepare_stage1_data.py")

        with patch.object(module, "prepare_stage1_artifacts", return_value=([{"id": "t1"}], [{"id": "d1"}])), \
            patch.object(module, "preview_rows", return_value="preview"), \
            patch.object(module, "write_jsonl") as mock_write, \
            patch.object(sys, "argv", ["prepare_stage1_data.py"]):
            with redirect_stdout(io.StringIO()):
                module.main()

        self.assertEqual(mock_write.call_count, 2)
        self.assertEqual(mock_write.call_args_list[0].args[0], "data/stage1/train.jsonl")
        self.assertEqual(mock_write.call_args_list[1].args[0], "data/stage1/dev200.jsonl")

    def test_prepare_eval_data_benchmarks_writes_gsm8k_and_math500(self) -> None:
        module = load_script_module("prepare_eval_data.py")

        with patch.object(module, "prepare_benchmark_artifact", side_effect=[[{"id": "g1"}], [{"id": "m1"}]]) as mock_prepare, \
            patch.object(module, "write_jsonl") as mock_write, \
            patch.object(sys, "argv", ["prepare_eval_data.py", "benchmarks"]):
            with redirect_stdout(io.StringIO()):
                module.main()

        self.assertEqual(mock_prepare.call_args_list[0].kwargs["dataset_name"], "gsm8k")
        self.assertEqual(mock_prepare.call_args_list[1].kwargs["dataset_name"], "HuggingFaceH4/MATH-500")
        self.assertEqual(mock_write.call_args_list[0].args[0], "data/eval/gsm8k_test.jsonl")
        self.assertEqual(mock_write.call_args_list[1].args[0], "data/eval/math500_test.jsonl")

    def test_prepare_eval_data_global_dev_writes_stratified_math500_dev_and_report(self) -> None:
        module = load_script_module("prepare_eval_data.py")

        with patch.object(
            module,
            "prepare_stratified_math500_dev_artifact",
            return_value=([{"id": "m1", "meta": {"source": "math500", "level": 1}}], {"sample_size": 200}),
        ) as mock_prepare, patch.object(module, "write_jsonl") as mock_write, patch(
            "pathlib.Path.write_text"
        ) as mock_report_write, patch.object(
            sys,
            "argv",
            ["prepare_eval_data.py", "global-dev"],
        ):
            with redirect_stdout(io.StringIO()):
                module.main()

        self.assertEqual(mock_prepare.call_args.kwargs["sample_size"], 200)
        self.assertEqual(mock_write.call_args_list[0].args[0], "data/eval/math500_dev200.jsonl")
        self.assertTrue(mock_report_write.called)

    def test_prepare_eval_data_aime_routes_year_argument(self) -> None:
        module = load_script_module("prepare_eval_data.py")

        with patch.object(module, "prepare_aime_eval_artifact", return_value=[{"id": "a1"}]) as mock_prepare, \
            patch.object(module, "write_aime_eval_artifact", return_value=Path("data/eval/aime25_test.jsonl")) as mock_write, \
            patch.object(module, "preview_rows", return_value="preview"), \
            patch.object(sys, "argv", ["prepare_eval_data.py", "aime", "--year", "25"]):
            with redirect_stdout(io.StringIO()):
                module.main()

        self.assertEqual(mock_prepare.call_args.kwargs["year"], 25)
        self.assertEqual(mock_write.call_args.kwargs["year"], 25)


if __name__ == "__main__":
    unittest.main()
