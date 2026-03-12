from __future__ import annotations

import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.answers import (
    are_equivalent,
    ensure_boxed_final_answer,
    extract_final_answer,
    extract_relaxed_final_answer,
    has_boxed_final_answer,
)
from rl.data import (
    clean_completion_for_protocol,
    canonical_problem_type,
    format_protocol_prompt,
    NUMINA_CATEGORY_TARGETS,
)
from rl.data import make_record


class AnswerUtilsTest(unittest.TestCase):
    def test_extract_boxed_answer(self) -> None:
        text = "Reasoning...\nFinal answer: \\boxed{42}"
        self.assertEqual(extract_final_answer(text), "42")

    def test_extract_plain_final_answer(self) -> None:
        text = "Reasoning...\nFinal answer: 1/2"
        self.assertEqual(extract_final_answer(text), "1/2")

    def test_extract_relaxed_answer(self) -> None:
        text = "Reasoning...\nThe answer is: \\frac{14}{3}"
        self.assertEqual(extract_relaxed_final_answer(text), "\\frac{14}{3}")

    def test_ensure_boxed_suffix(self) -> None:
        solution = "We compute the result.\nAnswer: 7"
        formatted = ensure_boxed_final_answer(solution, "7")
        self.assertIn("Final answer: \\boxed{7}", formatted)
        self.assertNotIn("Answer: 7", formatted)

    def test_equivalence_numeric(self) -> None:
        self.assertTrue(are_equivalent("0.5", "1/2"))

    def test_equivalence_latex_spacing(self) -> None:
        self.assertTrue(are_equivalent("The answer is: \\frac{1}{2}", "0.5"))

    def test_has_boxed_final_answer(self) -> None:
        self.assertTrue(has_boxed_final_answer("Final answer: \\boxed{42}"))

    def test_problem_type_normalization(self) -> None:
        self.assertEqual(canonical_problem_type("number_theory"), "Number Theory")
        self.assertEqual(canonical_problem_type("Logic / Puzzle"), "Logic and Puzzles")
        self.assertEqual(canonical_problem_type("unknown-tag"), "Other")

    def test_numina_targets_sum(self) -> None:
        self.assertEqual(sum(NUMINA_CATEGORY_TARGETS.values()), 3000)

    def test_make_record_prefers_answer_field(self) -> None:
        row = {
            "problem": "Compute 2+3.",
            "solution": "A long derivation without explicit final marker.",
            "answer": "5",
        }
        record = make_record(row, 0, "numinamath", include_solution=True)
        self.assertEqual(record["final_answer"], "5")
        self.assertIn("Final answer: \\boxed{5}", record["solution"])

    def test_make_record_builds_solution_from_answer_only(self) -> None:
        row = {
            "problem": "Compute 1/3.",
            "answer": "\\frac{1}{3}",
        }
        record = make_record(row, 0, "numinamath", include_solution=True)
        self.assertEqual(record["final_answer"], "\\frac{1}{3}")
        self.assertEqual(record["solution"], "Final answer: \\boxed{\\frac{1}{3}}")

    def test_make_record_extracts_gsm8k_final_answer(self) -> None:
        row = {
            "question": "Toy question",
            "answer": "Some reasoning\n#### 2450",
        }
        record = make_record(row, 0, "gsm8k", include_solution=True)
        self.assertEqual(record["final_answer"], "2450")
        self.assertEqual(record["solution"], "Final answer: \\boxed{2450}")

    def test_clean_completion_for_protocol_removes_solution_heading(self) -> None:
        solution = "## Solution.\n\nWe compute.\n\nFinal answer: \\boxed{42}"
        cleaned = clean_completion_for_protocol(solution)
        self.assertTrue(cleaned.startswith("We compute."))
        self.assertIn("Final answer: \\boxed{42}", cleaned)

    def test_clean_completion_for_protocol_removes_solution_numbering_heading(self) -> None:
        solution = "Solution 3\nReasoning line\n\nFinal answer: \\boxed{7}"
        cleaned = clean_completion_for_protocol(solution)
        self.assertTrue(cleaned.startswith("Reasoning line"))

    def test_format_protocol_prompt_requires_boxed_final_line(self) -> None:
        prompt = format_protocol_prompt("Compute 2+2.")
        self.assertIn("Question:\nCompute 2+2.", prompt)
        self.assertIn("Final answer: \\boxed{...}", prompt)


if __name__ == "__main__":
    unittest.main()
