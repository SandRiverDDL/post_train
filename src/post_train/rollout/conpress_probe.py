from __future__ import annotations

import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

from post_train.answers import evaluate_prediction, extract_boxed_content
from post_train.io import ensure_parent, read_jsonl, write_jsonl
from post_train.rollout.math_sft import DEFAULT_MATH_DATASET, build_math_query_pool


QUESTION_HEADER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:question|problem|q)\s*([0-9]+|[A-Z])\b[^\n]*"
)
ANSWER_HEADER_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:answer|solution)\s*([0-9]+|[A-Z])\b[^\n]*"
)
THINK_BLOCK_RE = re.compile(r"(?is)<think>(.*?)</think>")
BOXED_RE = re.compile(r"\\boxed\{")
XML_QUESTION_BLOCK_RE = re.compile(r"(?is)<question_([A-Z])>(.*?)</question_\1>")
XML_FINAL_ANSWER_RE = re.compile(r"(?is)([A-Z])\s*:\s*\\boxed\{")
VISUAL_MARKER_PATTERNS = [
    r"\[asy\]",
    r"\[/asy\]",
    r"\\begin\{asy\}",
    r"\\includegraphics",
    r"\bdiagram\b",
    r"figure shows",
    r"the figure",
    r"\[tikzpicture\]",
    r"\\begin\{tikzpicture\}",
    r"graph[^.]{0,80}shown below",
]
VISUAL_MARKER_RE = re.compile("|".join(VISUAL_MARKER_PATTERNS), re.IGNORECASE)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(ratio * len(ordered)) - 1))
    return int(ordered[index])


def _word_tokens(text: str) -> int:
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def _label_for_index(index: int) -> str:
    return chr(ord("A") + index - 1)


def has_visual_marker(question: str) -> bool:
    return bool(VISUAL_MARKER_RE.search(question))


def filter_visual_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = str(row.get("question", ""))
        if has_visual_marker(question):
            filtered.append(
                {
                    "index": index,
                    "id": str(row.get("id", "")),
                    "question": question,
                    "final_answer": str(row.get("final_answer", "")),
                    "meta": dict(row.get("meta", {})),
                }
            )
        else:
            kept.append(row)
    return kept, filtered


def _row_level(row: dict[str, Any]) -> int | None:
    value = row.get("level", row.get("meta", {}).get("level") if isinstance(row.get("meta"), dict) else None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_summary(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "index": index,
        "id": str(row.get("id", "")),
        "level": _row_level(row),
        "type": str(row.get("type", row.get("meta", {}).get("type", "")) if isinstance(row.get("meta"), dict) else row.get("type", "")),
        "question": str(row.get("question", "")),
        "final_answer": str(row.get("final_answer", "")),
        "meta": dict(row.get("meta", {})),
    }


def make_contiguous_question_packs(
    rows: list[dict[str, Any]],
    *,
    questions_per_prompt: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    pack_count = len(rows) // questions_per_prompt
    used_count = pack_count * questions_per_prompt
    packs = [
        rows[index : index + questions_per_prompt]
        for index in range(0, used_count, questions_per_prompt)
    ]
    dropped = [_row_summary(row, index) for index, row in enumerate(rows[used_count:], start=used_count)]
    return packs, dropped


def make_balanced_level_packs(
    rows: list[dict[str, Any]],
    *,
    questions_per_prompt: int,
    seed: int,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    if questions_per_prompt < 2:
        raise ValueError("questions_per_prompt 必须 >= 2。")

    indexed_rows = list(enumerate(rows))
    hard = [(index, row) for index, row in indexed_rows if _row_level(row) == 5]
    other = [(index, row) for index, row in indexed_rows if _row_level(row) != 5]
    rng = random.Random(seed)
    rng.shuffle(hard)
    rng.shuffle(other)

    packs: list[list[dict[str, Any]]] = []
    used_indices: set[int] = set()
    while hard and len(other) >= questions_per_prompt - 1:
        pack_items = [hard.pop()]
        for _ in range(questions_per_prompt - 1):
            pack_items.append(other.pop())
        rng.shuffle(pack_items)
        for index, _ in pack_items:
            used_indices.add(index)
        packs.append([row for _, row in pack_items])

    while len(other) >= questions_per_prompt:
        pack_items = [other.pop() for _ in range(questions_per_prompt)]
        rng.shuffle(pack_items)
        for index, _ in pack_items:
            used_indices.add(index)
        packs.append([row for _, row in pack_items])

    dropped = [
        _row_summary(row, index)
        for index, row in indexed_rows
        if index not in used_indices
    ]
    return packs, dropped


def build_conpress_prompt(rows: list[dict[str, Any]]) -> str:
    question_labels = ", ".join(f"Question {index}" for index in range(1, len(rows) + 1))
    parts = [
        "Please solve each question independently.",
        "Do not restate the problems and do not create new problems.",
        f"Use the same numbering in your response: {question_labels}.",
        "For each question, provide concise reasoning and end with Final answer: \\boxed{...}.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        parts.append(f"Question {index}: {row['question']}")
        parts.append("")
    return "\n".join(parts).rstrip()


def build_xml_oneshot_conpress_prompt(rows: list[dict[str, Any]]) -> str:
    if len(rows) > 26:
        raise ValueError("xml_oneshot prompt 最多支持 26 道题。")
    parts = [
        "Solve each question independently. Follow the output format exactly.",
        "",
        "Output rules:",
        "- Use exactly one <think>...</think> block.",
        "- Inside <think>, use one tagged block per question: <question_A>, <question_B>, <question_C>.",
        "- Each question block must end with exactly one line: Final answer: \\boxed{...}",
        "- After </think>, output exactly one <final_answers> block.",
        "- Do not output any other boxed answers.",
        "",
        "Example input:",
        "Question 1: What is 2+3?",
        "Question 2: What is the smallest positive square divisible by 2 and 3?",
        "Question 3: Solve x+4=10.",
        "",
        "Example output:",
        "<think>",
        "<question_A>",
        "2+3=5.",
        "Final answer: \\boxed{5}",
        "</question_A>",
        "",
        "<question_B>",
        "A number divisible by both 2 and 3 must be divisible by 6. The smallest square multiple is 6^2=36.",
        "Final answer: \\boxed{36}",
        "</question_B>",
        "",
        "<question_C>",
        "Subtract 4 from both sides to get x=6.",
        "Final answer: \\boxed{6}",
        "</question_C>",
        "</think>",
        "",
        "<final_answers>",
        "A: \\boxed{5}",
        "B: \\boxed{36}",
        "C: \\boxed{6}",
        "</final_answers>",
        "",
        "Now solve:",
    ]
    for index, row in enumerate(rows, start=1):
        parts.append(f"Question {index}: {row['question']}")
    return "\n".join(parts).rstrip()


def build_xml_tags_conpress_prompt(rows: list[dict[str, Any]]) -> str:
    if len(rows) > 26:
        raise ValueError("xml_tags prompt 最多支持 26 道题。")
    labels = [_label_for_index(index) for index in range(1, len(rows) + 1)]
    question_tags = ", ".join(f"<question_{label}>" for label in labels)
    final_lines = "\n".join(f"{label}: \\boxed{{...}}" for label in labels)
    parts = [
        "Solve each question independently. Follow the output format exactly.",
        "",
        "Output rules:",
        "- Begin your response with <think>.",
        f"- Inside <think>, use exactly these question blocks in order: {question_tags}.",
        "- End each question block with one line: Final answer: \\boxed{...}",
        "- Close the reasoning with </think>.",
        "- After </think>, output exactly one <final_answers> block with these lines:",
        final_lines,
        "- Do not copy these placeholders; replace every ... with the actual answer.",
        "- Do not output any boxed answers outside the question blocks and <final_answers>.",
        "",
        "Questions:",
    ]
    for index, row in enumerate(rows, start=1):
        parts.append(f"Question {index}: {row['question']}")
    return "\n".join(parts).rstrip()


def build_prompt(rows: list[dict[str, Any]], *, prompt_style: str) -> str:
    if prompt_style == "default":
        return build_conpress_prompt(rows)
    if prompt_style == "xml_oneshot":
        return build_xml_oneshot_conpress_prompt(rows)
    if prompt_style == "xml_tags":
        return build_xml_tags_conpress_prompt(rows)
    raise ValueError(f"未知 prompt_style：{prompt_style}")


def apply_chat_template(
    prompt: str,
    *,
    tokenizer_name: str,
    assistant_prefill: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if not tokenizer.chat_template:
        return prompt + (assistant_prefill or "")
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    return rendered + (assistant_prefill or "")


def make_question_packs(
    rows: list[dict[str, Any]],
    *,
    questions_per_prompt: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if questions_per_prompt < 2:
        raise ValueError("questions_per_prompt 必须 >= 2。")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    pack_count = len(shuffled) // questions_per_prompt
    packs: list[list[dict[str, Any]]] = []
    for pack_index in range(pack_count):
        start = pack_index * questions_per_prompt
        packs.append(shuffled[start : start + questions_per_prompt])
    return packs


def _anchor_matches(text: str) -> list[re.Match[str]]:
    answer_matches = list(ANSWER_HEADER_RE.finditer(text))
    if len(answer_matches) >= 2:
        return answer_matches
    matches = list(QUESTION_HEADER_RE.finditer(text))
    if len(matches) >= 2:
        return matches
    return answer_matches or matches


def _anchor_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    return ord(value.upper()) - ord("A") + 1


def _select_first_ordered_anchors(
    matches: list[re.Match[str]],
    *,
    questions_per_prompt: int,
) -> tuple[list[re.Match[str]], dict[str, Any]]:
    selected: list[re.Match[str]] = []
    duplicate_anchors: list[int] = []
    out_of_range_anchors: list[int] = []
    skipped_anchors: list[int] = []

    for start_index, candidate in enumerate(matches):
        if _anchor_number(candidate.group(1)) != 1:
            continue
        current: list[re.Match[str]] = []
        expected = 1
        for match in matches[start_index:]:
            number = _anchor_number(match.group(1))
            if number < 1 or number > questions_per_prompt:
                continue
            if number == expected:
                current.append(match)
                expected += 1
                if expected > questions_per_prompt:
                    selected = current
                    break
        if selected:
            break

    if not selected:
        seen: set[int] = set()
        for match in matches:
            number = _anchor_number(match.group(1))
            if number < 1 or number > questions_per_prompt:
                out_of_range_anchors.append(number)
                continue
            if number in seen:
                duplicate_anchors.append(number)
                continue
            selected.append(match)
            seen.add(number)

    selected_numbers = {_anchor_number(match.group(1)) for match in selected}
    for match in matches:
        number = _anchor_number(match.group(1))
        if number < 1 or number > questions_per_prompt:
            out_of_range_anchors.append(number)
            continue
        if number in selected_numbers and not any(match is selected_match for selected_match in selected):
            duplicate_anchors.append(number)
            continue
        if number not in selected_numbers:
            skipped_anchors.append(number)

    return selected, {
        "duplicate_anchors": duplicate_anchors,
        "out_of_range_anchors": out_of_range_anchors,
        "skipped_anchors": skipped_anchors,
    }


def split_by_question_anchors(text: str, *, questions_per_prompt: int) -> tuple[dict[int, str], dict[str, Any]]:
    all_matches = _anchor_matches(text)
    matches, selection_report = _select_first_ordered_anchors(all_matches, questions_per_prompt=questions_per_prompt)
    blocks: dict[int, str] = {}
    for match_index, match in enumerate(matches):
        number = _anchor_number(match.group(1))
        next_start = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(text)
        block = text[match.start() : next_start].strip()
        blocks[number] = block
    diagnostics = {
        "anchor_count": len(all_matches),
        "anchor_numbers": [_anchor_number(match.group(1)) for match in all_matches],
        "selected_anchor_numbers": [_anchor_number(match.group(1)) for match in matches],
        "missing_numbers": [number for number in range(1, questions_per_prompt + 1) if number not in blocks],
        "duplicate_anchors": selection_report["duplicate_anchors"],
        "out_of_range_anchors": selection_report["out_of_range_anchors"],
        "skipped_anchors": selection_report["skipped_anchors"],
        "used_answer_anchors": bool(all_matches and ANSWER_HEADER_RE.match(all_matches[0].group(0))),
    }
    return blocks, diagnostics


def split_by_xml_question_tags(text: str, *, questions_per_prompt: int) -> tuple[dict[int, str], dict[str, Any]]:
    matches = list(XML_QUESTION_BLOCK_RE.finditer(text))
    think_start = text.lower().find("<think>")
    first_question_start = matches[0].start() if matches else -1
    starts_like_xml = (
        think_start >= 0
        and first_question_start >= 0
        and first_question_start - think_start <= 500
    )
    if matches and not starts_like_xml:
        return {}, {
            "anchor_count": len(matches),
            "anchor_numbers": [_anchor_number(match.group(1)) for match in matches],
            "selected_anchor_numbers": [],
            "missing_numbers": list(range(1, questions_per_prompt + 1)),
            "duplicate_anchors": [],
            "out_of_range_anchors": [],
            "skipped_anchors": [],
            "used_answer_anchors": False,
            "used_xml_tags": False,
            "ignored_late_xml_tags": True,
            "ignored_late_xml_start": first_question_start,
        }
    blocks: dict[int, str] = {}
    duplicate_anchors: list[int] = []
    out_of_range_anchors: list[int] = []
    for match in matches:
        number = _anchor_number(match.group(1))
        if number < 1 or number > questions_per_prompt:
            out_of_range_anchors.append(number)
            continue
        if number in blocks:
            duplicate_anchors.append(number)
            continue
        blocks[number] = match.group(0).strip()
    diagnostics = {
        "anchor_count": len(matches),
        "anchor_numbers": [_anchor_number(match.group(1)) for match in matches],
        "selected_anchor_numbers": sorted(blocks),
        "missing_numbers": [number for number in range(1, questions_per_prompt + 1) if number not in blocks],
        "duplicate_anchors": duplicate_anchors,
        "out_of_range_anchors": out_of_range_anchors,
        "skipped_anchors": [],
        "used_answer_anchors": False,
        "used_xml_tags": bool(blocks),
    }
    return blocks, diagnostics


def _all_boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    for match in BOXED_RE.finditer(text):
        content = extract_boxed_content(text[match.start() :])
        if content:
            contents.append(content)
    return contents


def _last_boxed_content(text: str) -> str:
    contents = _all_boxed_contents(text)
    return contents[-1] if contents else ""


def _xml_final_answers(text: str, *, questions_per_prompt: int) -> list[str]:
    answers = [""] * questions_per_prompt
    for match in XML_FINAL_ANSWER_RE.finditer(text):
        number = _anchor_number(match.group(1))
        if number < 1 or number > questions_per_prompt:
            continue
        content = extract_boxed_content(text[match.start(0) + match.group(0).rfind("\\boxed{") :])
        if content:
            answers[number - 1] = content
    return answers if any(answers) else []


def _unordered_comma_equivalent(predicted: str, expected: str) -> bool:
    def normalize_items(value: str) -> list[str]:
        value = value.replace("\\,", ",")
        return [
            item.strip().replace(" ", "")
            for item in value.split(",")
            if item.strip().replace("\\", "")
        ]

    predicted_items = normalize_items(predicted)
    expected_items = normalize_items(expected)
    if len(predicted_items) <= 1 or len(predicted_items) != len(expected_items):
        return False
    return sorted(predicted_items) == sorted(expected_items)


def _answer_correct(predicted: str, expected: str) -> bool:
    if not predicted:
        return False
    verdict = evaluate_prediction(f"\\boxed{{{predicted}}}", expected, require_boxed=True)
    return bool(verdict["correct"]) or _unordered_comma_equivalent(predicted, expected)


def parse_conpress_output(
    text: str,
    pack_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    questions_per_prompt = len(pack_rows)
    think_blocks = THINK_BLOCK_RE.findall(text)
    blocks, anchor_report = split_by_xml_question_tags(text, questions_per_prompt=questions_per_prompt)
    if len(blocks) < questions_per_prompt:
        xml_anchor_report = anchor_report
        fallback_text = text[: int(xml_anchor_report["ignored_late_xml_start"])] if xml_anchor_report.get("ignored_late_xml_tags") else text
        blocks, anchor_report = split_by_question_anchors(fallback_text, questions_per_prompt=questions_per_prompt)
        anchor_report = {
            **anchor_report,
            "used_xml_tags": False,
            "ignored_late_xml_tags": bool(xml_anchor_report.get("ignored_late_xml_tags")),
            "ignored_late_xml_start": xml_anchor_report.get("ignored_late_xml_start"),
        }
    else:
        fallback_text = text
    global_boxed_answers = _all_boxed_contents(fallback_text)
    xml_answers = _xml_final_answers(fallback_text, questions_per_prompt=questions_per_prompt)
    used_xml_tags = bool(anchor_report.get("used_xml_tags"))
    tail_answers = (
        xml_answers
        if used_xml_tags and len(xml_answers) == questions_per_prompt and all(xml_answers)
        else global_boxed_answers[-questions_per_prompt:]
        if len(global_boxed_answers) >= questions_per_prompt
        else []
    )
    parsed_questions: list[dict[str, Any]] = []
    correct_count = 0
    parse_ok_count = 0
    boxed_count = 0
    multi_boxed_count = 0
    cross_talk_count = 0
    question_token_counts: list[int] = []

    for number, row in enumerate(pack_rows, start=1):
        block = blocks.get(number, "")
        boxed_occurrences = len(BOXED_RE.findall(block))
        block_boxed_answer = _last_boxed_content(block)
        paired_answer = tail_answers[number - 1] if len(tail_answers) == questions_per_prompt else ""
        boxed_answer = block_boxed_answer or paired_answer if used_xml_tags else paired_answer or block_boxed_answer
        verdict = evaluate_prediction(block, str(row.get("final_answer", "")), require_boxed=True) if block else {
            "boxed": False,
            "parse_ok": False,
            "parsed_answer": "",
            "correct": False,
        }
        if boxed_answer:
            verdict = {
                **verdict,
                "parsed_answer": boxed_answer,
                "parse_ok": True,
                "correct": _answer_correct(boxed_answer, str(row.get("final_answer", ""))),
            }
        other_question_refs = [
            ref
            for ref in range(1, questions_per_prompt + 1)
            if ref != number and re.search(rf"(?i)\b(?:question|problem|q)\s*{ref}\b", block)
        ]
        output_tokens = _word_tokens(block)
        question_token_counts.append(output_tokens)
        boxed = bool(boxed_answer) or bool(verdict["boxed"])
        parse_ok = bool(verdict["parse_ok"])
        correct = bool(verdict["correct"])
        boxed_count += int(boxed)
        parse_ok_count += int(parse_ok)
        correct_count += int(correct)
        multi_boxed_count += int(boxed_occurrences > 1)
        cross_talk_count += int(bool(other_question_refs))
        parsed_questions.append(
            {
                "question_index": number,
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "expected_answer": str(row.get("final_answer", "")),
                "block": block,
                "output_tokens": output_tokens,
                "boxed_occurrences": boxed_occurrences,
                "block_boxed_answer": block_boxed_answer,
                "paired_answer_source": (
                    "block_last_boxed"
                    if used_xml_tags and block_boxed_answer
                    else "tail_boxed"
                    if paired_answer
                    else "block_last_boxed"
                    if block_boxed_answer
                    else ""
                ),
                "boxed": boxed,
                "parse_ok": parse_ok,
                "predicted_answer": str(verdict.get("parsed_answer", "")),
                "correct": correct,
                "other_question_refs": other_question_refs,
                "meta": dict(row.get("meta", {})),
            }
        )

    reasoning_split_ok = len(anchor_report["missing_numbers"]) == 0
    answer_pairing_ok = len(tail_answers) == questions_per_prompt or boxed_count == questions_per_prompt
    format_ok = reasoning_split_ok and answer_pairing_ok and not anchor_report["out_of_range_anchors"]
    exact_n_boxed = len(BOXED_RE.findall(text)) == questions_per_prompt
    return {
        "format_ok": format_ok,
        "reasoning_split_ok": reasoning_split_ok,
        "answer_pairing_ok": answer_pairing_ok,
        "exact_n_boxed": exact_n_boxed,
        "all_correct": correct_count == questions_per_prompt,
        "all_parse_ok": parse_ok_count == questions_per_prompt,
        "all_boxed": boxed_count == questions_per_prompt,
        "question_count": questions_per_prompt,
        "boxed_count": boxed_count,
        "parse_ok_count": parse_ok_count,
        "correct_count": correct_count,
        "multi_boxed_count": multi_boxed_count,
        "cross_talk_count": cross_talk_count,
        "output_tokens": _word_tokens(text),
        "avg_question_tokens": sum(question_token_counts) / len(question_token_counts) if question_token_counts else 0.0,
        "think_block_count": len(think_blocks),
        "global_boxed_count": len(global_boxed_answers),
        "tail_boxed_answers": tail_answers,
        "xml_final_answers": xml_answers,
        "anchor_report": anchor_report,
        "questions": parsed_questions,
    }


def run_conpress_probe(
    *,
    model: str,
    output_dir: str | Path,
    dataset_name: str = DEFAULT_MATH_DATASET,
    split: str = "train",
    sample_size: int = 99,
    questions_per_prompt: int = 3,
    samples_per_prompt: int = 1,
    levels: list[int] | None = None,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 32,
    max_new_tokens: int = 8192,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.85,
    enforce_eager: bool = False,
    cache_dir: str | None = None,
    query_pool_path: str | Path | None = None,
    num_shards: int = 1,
    shard_index: int | None = None,
    trust_remote_code: bool = True,
    use_chat_template: bool = True,
    chat_template_enable_thinking: bool | None = None,
    tokenizer_name: str | None = None,
    exclude_visual: bool = False,
    pack_strategy: str | None = None,
    prompt_style: str = "default",
    assistant_prefill: str | None = None,
) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    base = Path(output_dir)
    if query_pool_path is not None:
        query_rows = [dict(row) for row in read_jsonl(query_pool_path)]
        visual_filtered_rows: list[dict[str, Any]] = []
        if exclude_visual:
            query_rows, visual_filtered_rows = filter_visual_rows(query_rows)
        query_report = {
            "dataset_name": dataset_name,
            "split": split,
            "query_pool_path": str(query_pool_path),
            "exclude_visual": exclude_visual,
            "filtered_visual_count": len(visual_filtered_rows),
            "filtered_visual_rows": visual_filtered_rows,
            "visual_marker_patterns": VISUAL_MARKER_PATTERNS,
            "seed": seed,
        }
    else:
        question_sample_size = max(questions_per_prompt, sample_size)
        query_rows, query_report = build_math_query_pool(
            dataset_name=dataset_name,
            split=split,
            sample_size=question_sample_size,
            seed=seed,
            levels=levels,
            cache_dir=cache_dir,
        )
        visual_filtered_rows = []
        if exclude_visual:
            query_rows, visual_filtered_rows = filter_visual_rows(query_rows)
        query_report = {
            **query_report,
            "exclude_visual": exclude_visual,
            "filtered_visual_count": len(visual_filtered_rows),
            "filtered_visual_rows": visual_filtered_rows,
            "visual_marker_patterns": VISUAL_MARKER_PATTERNS,
        }
    resolved_pack_strategy = pack_strategy or ("contiguous" if query_pool_path is not None else "shuffle")
    if resolved_pack_strategy == "contiguous":
        packs, pack_dropped_rows = make_contiguous_question_packs(query_rows, questions_per_prompt=questions_per_prompt)
    elif resolved_pack_strategy == "shuffle":
        packs = make_question_packs(query_rows, questions_per_prompt=questions_per_prompt, seed=seed + 1)
        used_count = len(packs) * questions_per_prompt
        pack_dropped_rows = [_row_summary(row, index) for index, row in enumerate(query_rows[used_count:], start=used_count)]
    elif resolved_pack_strategy == "balanced_level":
        packs, pack_dropped_rows = make_balanced_level_packs(
            query_rows,
            questions_per_prompt=questions_per_prompt,
            seed=seed + 1,
        )
    else:
        raise ValueError(f"未知 pack_strategy：{resolved_pack_strategy}")
    question_sample_size = len(packs) * questions_per_prompt
    query_rows = [row for pack in packs for row in pack]
    query_report = {
        **query_report,
        "sampled_queries": len(query_rows),
        "candidate_queries": len(query_rows) + len(pack_dropped_rows),
        "pack_strategy": resolved_pack_strategy,
        "pack_dropped_count": len(pack_dropped_rows),
        "pack_dropped_rows": pack_dropped_rows,
    }
    total_pack_count = len(packs)
    if num_shards < 1:
        raise ValueError("num_shards 必须 >= 1。")
    if shard_index is not None:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"shard_index 必须在 [0, {num_shards})，实际为 {shard_index}。")
        packs = [pack for index, pack in enumerate(packs) if index % num_shards == shard_index]
    raw_prompts = [build_prompt(pack, prompt_style=prompt_style) for pack in packs]
    template_tokenizer_name = tokenizer_name or model
    prompts = [
        apply_chat_template(
            prompt,
            tokenizer_name=template_tokenizer_name,
            assistant_prefill=assistant_prefill,
            enable_thinking=chat_template_enable_thinking,
        )
        if use_chat_template
        else prompt + (assistant_prefill or "")
        for prompt in raw_prompts
    ]
    prompt_rows = [
        {
            "pack_id": f"pack_{index}",
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "assistant_prefill": assistant_prefill,
            "questions": pack,
        }
        for index, (raw_prompt, prompt, pack) in enumerate(zip(raw_prompts, prompts, packs, strict=True))
    ]
    write_jsonl(base / "prompts.jsonl", prompt_rows)

    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
    )
    sampling_params = SamplingParams(
        n=samples_per_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop=["</s>", "<|im_end|>"],
    )

    started_at = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    total_seconds = time.perf_counter() - started_at

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    output_tokens: list[int] = []
    question_tokens: list[int] = []
    format_ok = 0
    reasoning_split_ok = 0
    answer_pairing_ok = 0
    exact_n_boxed = 0
    all_correct = 0
    all_parse_ok = 0
    total_generations = 0
    total_questions = 0
    total_correct = 0
    total_parse_ok = 0
    total_boxed = 0
    total_multi_boxed = 0
    total_cross_talk = 0

    for pack_index, (pack, prompt, output) in enumerate(zip(packs, prompts, outputs, strict=True)):
        responses: list[dict[str, Any]] = []
        for sample_index, candidate in enumerate(output.outputs):
            generated_text = candidate.text
            text = (assistant_prefill or "") + generated_text
            parsed = parse_conpress_output(text, pack)
            total_generations += 1
            total_questions += parsed["question_count"]
            total_correct += parsed["correct_count"]
            total_parse_ok += parsed["parse_ok_count"]
            total_boxed += parsed["boxed_count"]
            total_multi_boxed += parsed["multi_boxed_count"]
            total_cross_talk += parsed["cross_talk_count"]
            format_ok += int(parsed["format_ok"])
            reasoning_split_ok += int(parsed["reasoning_split_ok"])
            answer_pairing_ok += int(parsed["answer_pairing_ok"])
            exact_n_boxed += int(parsed["exact_n_boxed"])
            all_correct += int(parsed["all_correct"])
            all_parse_ok += int(parsed["all_parse_ok"])
            output_tokens.append(int(parsed["output_tokens"]))
            question_tokens.extend(int(question["output_tokens"]) for question in parsed["questions"])
            responses.append({"sample_index": sample_index, "text": text, "generated_text": generated_text})
            parsed_rows.append(
                {
                    "pack_id": f"pack_{pack_index}",
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "raw_text": text,
                    **parsed,
                }
            )
        raw_rows.append(
            {
                "pack_id": f"pack_{pack_index}",
                "prompt": prompt,
                "questions": pack,
                "responses": responses,
            }
        )

    raw_path = base / "raw_outputs.jsonl"
    parsed_path = base / "parsed_outputs.jsonl"
    report_path = base / "report.json"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(parsed_path, parsed_rows)

    report = {
        "model": model,
        "dataset": query_report,
        "params": {
            "questions_per_prompt": questions_per_prompt,
            "sample_size": question_sample_size,
            "pack_count": len(packs),
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "seed": seed,
            "use_chat_template": use_chat_template,
            "chat_template_enable_thinking": chat_template_enable_thinking,
            "tokenizer_name": template_tokenizer_name,
            "pack_strategy": resolved_pack_strategy,
            "prompt_style": prompt_style,
            "assistant_prefill": assistant_prefill,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "total_pack_count": total_pack_count,
        },
        "metrics": {
            "generation_count": total_generations,
            "total_questions": total_questions,
            "total_seconds": total_seconds,
            "generations_per_second": total_generations / total_seconds if total_seconds > 0 else 0.0,
            "format_ok_rate": format_ok / total_generations if total_generations else 0.0,
            "reasoning_split_ok_rate": reasoning_split_ok / total_generations if total_generations else 0.0,
            "answer_pairing_ok_rate": answer_pairing_ok / total_generations if total_generations else 0.0,
            "exact_n_boxed_rate": exact_n_boxed / total_generations if total_generations else 0.0,
            "all_correct_rate": all_correct / total_generations if total_generations else 0.0,
            "all_parse_ok_rate": all_parse_ok / total_generations if total_generations else 0.0,
            "question_correct_rate": total_correct / total_questions if total_questions else 0.0,
            "question_parse_ok_rate": total_parse_ok / total_questions if total_questions else 0.0,
            "question_boxed_rate": total_boxed / total_questions if total_questions else 0.0,
            "multi_boxed_block_rate": total_multi_boxed / total_questions if total_questions else 0.0,
            "cross_talk_block_rate": total_cross_talk / total_questions if total_questions else 0.0,
            "avg_output_tokens": statistics.mean(output_tokens) if output_tokens else 0.0,
            "p50_output_tokens": _percentile(output_tokens, 0.50),
            "p90_output_tokens": _percentile(output_tokens, 0.90),
            "p95_output_tokens": _percentile(output_tokens, 0.95),
            "max_output_tokens": max(output_tokens) if output_tokens else 0,
            "avg_question_tokens": statistics.mean(question_tokens) if question_tokens else 0.0,
            "p50_question_tokens": _percentile(question_tokens, 0.50),
            "p90_question_tokens": _percentile(question_tokens, 0.90),
            "p95_question_tokens": _percentile(question_tokens, 0.95),
            "max_question_tokens": max(question_tokens) if question_tokens else 0,
        },
        "outputs": {
            "prompts": str(base / "prompts.jsonl"),
            "raw_outputs": str(raw_path),
            "parsed_outputs": str(parsed_path),
            "report": str(report_path),
        },
    }
    _write_json(report_path, report)
    return report
