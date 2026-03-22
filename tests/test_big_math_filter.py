from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import filter_big_math_dataset
from rl.data import count_latex_markers, is_allowed_short_answer, looks_like_proof_prompt


class BigMathFilterTest(unittest.TestCase):
    def test_count_latex_markers_counts_commands_and_delimiters(self) -> None:
        text = r"Solve \(x\) where \frac{1}{2} + \sqrt{y}"
        self.assertEqual(count_latex_markers(text), 4)

    def test_answer_shape_allows_short_scalar_forms_and_blocks_sets(self) -> None:
        self.assertTrue(is_allowed_short_answer("42"))
        self.assertTrue(is_allowed_short_answer("-3.5"))
        self.assertTrue(is_allowed_short_answer("7/8"))
        self.assertTrue(is_allowed_short_answer("2x+1"))
        self.assertFalse(is_allowed_short_answer(r"\{1,2\}"))
        self.assertFalse(is_allowed_short_answer("[1,2)"))
        self.assertFalse(is_allowed_short_answer("x = 3"))

    def test_proof_prompt_uses_keyword_heuristics(self) -> None:
        self.assertTrue(looks_like_proof_prompt("Prove that the sequence converges."))
        self.assertTrue(looks_like_proof_prompt("证明这个结论成立。"))
        self.assertFalse(looks_like_proof_prompt("Solve for x in 2x+3=7."))

    def test_main_filters_and_keeps_middle_sixty_percent(self) -> None:
        rows = [
            {
                "source": "orca_math",
                "domain": ["Mathematics -> Algebra -> Simple Equations"],
                "prompt": "Solve x+1=2.",
                "solution": "1",
                "llama8b_solve_rate": 0.1,
            },
            {
                "source": "big_math",
                "domain": ["Mathematics -> Prealgebra -> Word Problems"],
                "prompt": "A store sold 12 pens and 3 pencils. How many items?",
                "solution": "15",
                "llama8b_solve_rate": 0.3,
            },
            {
                "source": "math",
                "domain": ["Mathematics -> Applied Mathematics -> Math Word Problems"],
                "prompt": "If 3 bags have 4 apples each, how many apples are there?",
                "solution": "12",
                "llama8b_solve_rate": 0.5,
            },
            {
                "source": "orca_math",
                "domain": ["Mathematics -> Algebra -> Equations and Inequalities"],
                "prompt": "Find x if 2x=18.",
                "solution": "9",
                "llama8b_solve_rate": 0.7,
            },
            {
                "source": "orca_math",
                "domain": ["Mathematics -> Algebra -> Equations and Inequalities"],
                "prompt": "What is 10 divided by 2?",
                "solution": "5",
                "llama8b_solve_rate": 0.9,
            },
            {
                "source": "aops_forum",
                "domain": ["Mathematics -> Algebra -> Simple Equations"],
                "prompt": "Solve x+2=5.",
                "solution": "3",
                "llama8b_solve_rate": 0.4,
            },
            {
                "source": "orca_math",
                "domain": ["Mathematics -> Algebra -> Simple Equations"],
                "prompt": "Prove that x+1 is positive.",
                "solution": "1",
                "llama8b_solve_rate": 0.4,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "big_math.jsonl"
            summary_path = Path(tmpdir) / "big_math_summary.json"

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "filter_big_math_dataset.py",
                        "--output",
                        str(output_path),
                        "--summary-output",
                        str(summary_path),
                    ],
                ),
                patch.object(filter_big_math_dataset, "load_dataset", return_value=rows),
            ):
                filter_big_math_dataset.main()

            kept_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(kept_rows), 3)
            self.assertEqual([row["final_answer"] for row in kept_rows], ["15", "12", "9"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["kept_count"], 3)
            self.assertEqual(summary["dropped_by_reason"]["source_excluded"], 1)
            self.assertEqual(summary["dropped_by_reason"]["proof_like_prompt"], 1)


if __name__ == "__main__":
    unittest.main()
