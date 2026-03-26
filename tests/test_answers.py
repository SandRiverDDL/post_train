from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.answers import (
    are_equivalent,
    ensure_boxed_final_answer,
    extract_final_answer,
    extract_relaxed_final_answer,
    has_boxed_final_answer,
)


class AnswerUtilsTest(unittest.TestCase):
    def test_extract_boxed_answer(self) -> None:
        text = "Reasoning...\n\\boxed{42}"
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
        self.assertIn("\\boxed{7}", formatted)
        self.assertNotIn("Answer: 7", formatted)

    def test_equivalence_numeric(self) -> None:
        self.assertTrue(are_equivalent("0.5", "1/2"))

    def test_equivalence_latex_spacing(self) -> None:
        self.assertTrue(are_equivalent("The answer is: \\frac{1}{2}", "0.5"))

    def test_has_boxed_final_answer(self) -> None:
        self.assertTrue(has_boxed_final_answer("\\boxed{42}"))


if __name__ == "__main__":
    unittest.main()
