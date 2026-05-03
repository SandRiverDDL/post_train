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

    def test_extract_final_answer_uses_last_boxed_answer(self) -> None:
        text = "A wrong intermediate result is \\boxed{0}.\nFinal answer: \\boxed{42}"
        self.assertEqual(extract_final_answer(text), "42")

    def test_extract_final_answer_skips_unclosed_boxed_prefix(self) -> None:
        text = "Bad prefix \\boxed{not closed\nFinal answer: \\boxed{42}"
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

    def test_equivalence_fraction_command_variants(self) -> None:
        self.assertTrue(are_equivalent("\\dfrac{5}{13}", "\\frac{5}{13}"))
        self.assertTrue(are_equivalent("\\tfrac{5}{13}", "\\frac{5}{13}"))
        self.assertTrue(are_equivalent("\\cfrac{5}{13}", "\\frac{5}{13}"))

    def test_equivalence_latex_spacing_commands(self) -> None:
        self.assertTrue(are_equivalent("\\left(1\\; +\\! 2\\right)", "(1+2)"))

    def test_equivalence_exact_latex_match_without_parser_support(self) -> None:
        self.assertTrue(are_equivalent("-\\frac{\\pi}{6}", "-\\frac{\\pi}{6}"))
        self.assertTrue(are_equivalent("\\cot x", "\\cot x"))
        self.assertTrue(are_equivalent("[-2, 7]", "[-2, 7]"))

    def test_has_boxed_final_answer(self) -> None:
        self.assertTrue(has_boxed_final_answer("\\boxed{42}"))


if __name__ == "__main__":
    unittest.main()
