from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from post_train.rollout.conpress_probe import (
    apply_chat_template,
    build_conpress_prompt,
    build_xml_tags_conpress_prompt,
    build_xml_oneshot_conpress_prompt,
    filter_visual_rows,
    make_balanced_level_packs,
    make_question_packs,
    parse_conpress_output,
    run_conpress_probe,
    split_by_question_anchors,
)


class ConPressProbeTest(unittest.TestCase):
    def test_apply_chat_template_passes_enable_thinking_false(self) -> None:
        class FakeTokenizer:
            chat_template = "fake"

            def __init__(self) -> None:
                self.kwargs = {}

            def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs) -> str:
                self.kwargs = kwargs
                return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        fake_tokenizer = FakeTokenizer()

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs) -> FakeTokenizer:
                return fake_tokenizer

        sys.modules["transformers"] = type(
            "FakeTransformers",
            (),
            {"AutoTokenizer": FakeAutoTokenizer},
        )
        try:
            rendered = apply_chat_template("1+1?", tokenizer_name="fake", enable_thinking=False)
        finally:
            sys.modules.pop("transformers", None)

        self.assertFalse(fake_tokenizer.kwargs["enable_thinking"])
        self.assertIn("<think>", rendered)

    def test_split_by_question_anchors_uses_question_numbers(self) -> None:
        text = "Question 1: solve one\nFinal answer: \\boxed{1}\n\nQuestion 2: solve two\nFinal answer: \\boxed{2}"

        blocks, report = split_by_question_anchors(text, questions_per_prompt=2)

        self.assertEqual(set(blocks), {1, 2})
        self.assertEqual(report["missing_numbers"], [])
        self.assertIn("\\boxed{1}", blocks[1])
        self.assertIn("\\boxed{2}", blocks[2])

    def test_parse_keeps_multiple_boxed_inside_same_block(self) -> None:
        rows = [
            {"id": "q1", "question": "1+0?", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "1+1?", "final_answer": "2", "meta": {}},
        ]
        text = (
            "Question 1: First I think \\boxed{0} is wrong. Final answer: \\boxed{1}\n\n"
            "Question 2: Final answer: \\boxed{2}"
        )

        parsed = parse_conpress_output(text, rows)

        self.assertTrue(parsed["format_ok"])
        self.assertEqual(parsed["correct_count"], 2)
        self.assertEqual(parsed["multi_boxed_count"], 1)
        self.assertEqual(parsed["questions"][0]["predicted_answer"], "1")

    def test_missing_question_is_not_format_ok(self) -> None:
        rows = [
            {"id": "q1", "question": "1+0?", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "1+1?", "final_answer": "2", "meta": {}},
            {"id": "q3", "question": "1+2?", "final_answer": "3", "meta": {}},
        ]
        text = "Question 1: Final answer: \\boxed{1}\n\nQuestion 3: Final answer: \\boxed{3}"

        parsed = parse_conpress_output(text, rows)

        self.assertFalse(parsed["format_ok"])
        self.assertEqual(parsed["anchor_report"]["missing_numbers"], [2])

    def test_make_question_packs_drops_incomplete_tail(self) -> None:
        rows = [{"id": str(index), "question": str(index), "final_answer": str(index)} for index in range(7)]

        packs = make_question_packs(rows, questions_per_prompt=3, seed=1)

        self.assertEqual(len(packs), 2)
        self.assertTrue(all(len(pack) == 3 for pack in packs))

    def test_build_prompt_contains_explicit_question_labels(self) -> None:
        rows = [
            {"question": "A?"},
            {"question": "B?"},
            {"question": "C?"},
        ]

        prompt = build_conpress_prompt(rows)

        self.assertIn("Question 1: A?", prompt)
        self.assertIn("Question 2: B?", prompt)
        self.assertIn("Question 3: C?", prompt)
        self.assertIn("Use the same numbering", prompt)
        self.assertIn("Final answer: \\boxed{...}", prompt)

    def test_build_prompt_uses_dynamic_question_labels(self) -> None:
        rows = [
            {"question": "A?"},
            {"question": "B?"},
        ]

        prompt = build_conpress_prompt(rows)

        self.assertIn("Question 1, Question 2.", prompt)
        self.assertNotIn("Question 3", prompt)

    def test_build_xml_oneshot_prompt_contains_tags_and_example(self) -> None:
        rows = [
            {"question": "A?"},
            {"question": "B?"},
            {"question": "C?"},
        ]

        prompt = build_xml_oneshot_conpress_prompt(rows)

        self.assertIn("<question_A>", prompt)
        self.assertIn("<final_answers>", prompt)
        self.assertIn("A: \\boxed{5}", prompt)
        self.assertIn("Question 1: A?", prompt)

    def test_build_xml_tags_prompt_has_no_example_answers(self) -> None:
        rows = [
            {"question": "A?"},
            {"question": "B?"},
            {"question": "C?"},
        ]

        prompt = build_xml_tags_conpress_prompt(rows)

        self.assertIn("<question_A>", prompt)
        self.assertIn("<final_answers>", prompt)
        self.assertIn("A: \\boxed{...}", prompt)
        self.assertNotIn("2+3=5", prompt)
        self.assertNotIn("A: \\boxed{5}", prompt)
        self.assertIn("Question 1: A?", prompt)

    def test_split_accepts_answer_letters(self) -> None:
        text = "Answer A: Final answer: \\boxed{1}\n\nAnswer B: Final answer: \\boxed{2}"

        blocks, report = split_by_question_anchors(text, questions_per_prompt=2)

        self.assertEqual(set(blocks), {1, 2})
        self.assertEqual(report["missing_numbers"], [])
        self.assertTrue(report["used_answer_anchors"])

    def test_parse_problem_anchors_with_tail_boxed_summary(self) -> None:
        rows = [
            {"id": "q1", "question": "det?", "final_answer": "46", "meta": {}},
            {"id": "q2", "question": "values?", "final_answer": "5,\\,-11", "meta": {}},
            {"id": "q3", "question": "crt?", "final_answer": "2519", "meta": {}},
        ]
        text = (
            "<think>\n"
            "**Problem A:** compute determinant.\n"
            "**Problem B:** solve the parameter values.\n"
            "**Problem C:** solve the congruences.\n"
            "Problem A: \\boxed{46}\n"
            "Problem B: \\boxed{-11,5}\n"
            "Problem C: \\boxed{2519}\n"
            "</think>"
        )

        parsed = parse_conpress_output(text, rows)

        self.assertTrue(parsed["reasoning_split_ok"])
        self.assertTrue(parsed["answer_pairing_ok"])
        self.assertEqual(parsed["correct_count"], 3)
        self.assertEqual(parsed["questions"][1]["predicted_answer"], "-11,5")

    def test_parse_prefers_xml_question_tags_and_final_answers(self) -> None:
        rows = [
            {"id": "q1", "question": "one", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "two", "final_answer": "2", "meta": {}},
            {"id": "q3", "question": "three", "final_answer": "3", "meta": {}},
        ]
        text = (
            "<think>\n"
            "<question_A>\nQuestion 99 distractor. Final answer: \\boxed{1}\n</question_A>\n"
            "<question_B>\nFinal answer: \\boxed{2}\n</question_B>\n"
            "<question_C>\nFinal answer: \\boxed{3}\n</question_C>\n"
            "</think>\n"
            "<final_answers>\n"
            "A: \\boxed{5}\n"
            "B: \\boxed{36}\n"
            "C: \\boxed{6}\n"
            "</final_answers>"
        )

        parsed = parse_conpress_output(text, rows)

        self.assertTrue(parsed["anchor_report"]["used_xml_tags"])
        self.assertTrue(parsed["answer_pairing_ok"])
        self.assertEqual(parsed["correct_count"], 3)
        self.assertEqual(parsed["questions"][0]["predicted_answer"], "1")

    def test_parse_ignores_late_copied_xml_example(self) -> None:
        rows = [
            {"id": "q1", "question": "one", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "two", "final_answer": "2", "meta": {}},
            {"id": "q3", "question": "three", "final_answer": "3", "meta": {}},
        ]
        text = (
            "<think>\n"
            "Question 1: Final answer: \\boxed{1}\n"
            "Question 2: Final answer: \\boxed{2}\n"
            "Question 3: Final answer: \\boxed{3}\n"
            f"{'After solving, the model continues reasoning. ' * 20}\n"
            "Then the model copied the example:\n"
            "<question_A>Final answer: \\boxed{5}</question_A>\n"
            "<question_B>Final answer: \\boxed{36}</question_B>\n"
            "<question_C>Final answer: \\boxed{6}</question_C>\n"
            "</think>"
        )

        parsed = parse_conpress_output(text, rows)

        self.assertFalse(parsed["anchor_report"]["used_xml_tags"])
        self.assertEqual(parsed["correct_count"], 3)
        self.assertEqual(parsed["questions"][0]["predicted_answer"], "1")

    def test_run_probe_uses_existing_query_pool_for_pack_order(self) -> None:
        rows = [
            {"id": "q1", "question": "1+0?", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "1+1?", "final_answer": "2", "meta": {}},
        ]

        class FakeCandidate:
            text = "Question 1: Final answer: \\boxed{1}\nQuestion 2: Final answer: \\boxed{2}"

        class FakeOutput:
            outputs = [FakeCandidate()]

        class FakeLLM:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def generate(self, prompts, sampling_params, use_tqdm):
                return [FakeOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, *args, **kwargs) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            query_pool = Path(tmpdir) / "query_pool.jsonl"
            query_pool.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
            sys.modules["vllm"] = type(
                "FakeVllm",
                (),
                {"LLM": FakeLLM, "SamplingParams": FakeSamplingParams},
            )
            try:
                report = run_conpress_probe(
                    model="fake",
                    output_dir=Path(tmpdir) / "out",
                    query_pool_path=query_pool,
                    questions_per_prompt=2,
                    sample_size=99,
                    use_chat_template=False,
                )
            finally:
                sys.modules.pop("vllm", None)

        self.assertEqual(report["dataset"]["query_pool_path"], str(query_pool))
        self.assertEqual(report["dataset"]["sampled_queries"], 2)
        self.assertEqual(report["metrics"]["question_correct_rate"], 1.0)

    def test_run_probe_prepends_assistant_prefill_to_parsed_text(self) -> None:
        rows = [
            {"id": "q1", "question": "1+0?", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "1+1?", "final_answer": "2", "meta": {}},
        ]

        class FakeCandidate:
            text = " Final answer: \\boxed{1}\nQuestion 2: Final answer: \\boxed{2}"

        class FakeOutput:
            outputs = [FakeCandidate()]

        class FakeLLM:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def generate(self, prompts, sampling_params, use_tqdm):
                return [FakeOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, *args, **kwargs) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            query_pool = Path(tmpdir) / "query_pool.jsonl"
            query_pool.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")
            sys.modules["vllm"] = type(
                "FakeVllm",
                (),
                {"LLM": FakeLLM, "SamplingParams": FakeSamplingParams},
            )
            try:
                report = run_conpress_probe(
                    model="fake",
                    output_dir=Path(tmpdir) / "out",
                    query_pool_path=query_pool,
                    questions_per_prompt=2,
                    sample_size=99,
                    use_chat_template=False,
                    assistant_prefill="<think>\nQuestion 1:",
                )
            finally:
                sys.modules.pop("vllm", None)

            self.assertEqual(report["metrics"]["question_correct_rate"], 1.0)
            parsed = __import__("json").loads((Path(report["outputs"]["parsed_outputs"]).read_text(encoding="utf-8").splitlines()[0]))
            self.assertTrue(parsed["raw_text"].startswith("<think>\nQuestion 1:"))

    def test_exclude_visual_filters_query_pool_before_packing(self) -> None:
        rows = [
            {"id": "q1", "question": "1+0?", "final_answer": "1", "meta": {}},
            {"id": "q2", "question": "[asy] draw((0,0)--(1,1)); [/asy] Find x.", "final_answer": "2", "meta": {}},
            {"id": "q3", "question": "The figure shows a square. Find y.", "final_answer": "3", "meta": {}},
            {"id": "q4", "question": "2+2?", "final_answer": "4", "meta": {}},
        ]

        kept, filtered = filter_visual_rows(rows)

        self.assertEqual([row["id"] for row in kept], ["q1", "q4"])
        self.assertEqual([row["id"] for row in filtered], ["q2", "q3"])

    def test_balanced_level_packs_avoid_multiple_level5_when_possible(self) -> None:
        rows = [
            {"id": "h1", "question": "h1", "final_answer": "1", "meta": {"level": 5}},
            {"id": "h2", "question": "h2", "final_answer": "1", "meta": {"level": 5}},
            {"id": "h3", "question": "h3", "final_answer": "1", "meta": {"level": 5}},
            {"id": "e1", "question": "e1", "final_answer": "1", "meta": {"level": 2}},
            {"id": "e2", "question": "e2", "final_answer": "1", "meta": {"level": 3}},
            {"id": "e3", "question": "e3", "final_answer": "1", "meta": {"level": 1}},
            {"id": "e4", "question": "e4", "final_answer": "1", "meta": {"level": 4}},
        ]

        packs, dropped = make_balanced_level_packs(rows, questions_per_prompt=3, seed=1)

        self.assertEqual(len(packs), 2)
        for pack in packs:
            self.assertLessEqual(sum(1 for row in pack if row["meta"]["level"] == 5), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["level"], 5)


if __name__ == "__main__":
    unittest.main()
