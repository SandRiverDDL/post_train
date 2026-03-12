from __future__ import annotations

import re
from typing import Any

from rl.answers import ensure_boxed_final_answer, extract_relaxed_final_answer


QUESTION_KEYS = ("problem", "query", "question", "prompt")
SOLUTION_KEYS = ("response", "solution", "completion")
FINAL_ANSWER_KEYS = ("answer", "final_answer")
ID_KEYS = ("id", "problem_id", "uuid")
PROBLEM_TYPE_KEYS = ("problem_type", "type", "subject", "category")
GSM8K_ANSWER_RE = re.compile(r"####\s*(.+)$", re.DOTALL)

NUMINA_CATEGORY_TARGETS = {
    "Algebra": 750,
    "Geometry": 600,
    "Number Theory": 500,
    "Combinatorics": 450,
    "Logic and Puzzles": 250,
    "Calculus": 180,
    "Inequalities": 180,
    "Other": 90,
}

CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Algebra", ("algebra",)),
    ("Geometry", ("geometry", "geometric")),
    ("Number Theory", ("number theory", "number_theory", "nt")),
    ("Combinatorics", ("combinatorics", "counting", "pigeonhole")),
    ("Logic and Puzzles", ("logic and puzzles", "logic", "puzzle")),
    ("Calculus", ("calculus", "derivative", "integral")),
    ("Inequalities", ("inequalities", "inequality")),
    ("Other", ("other",)),
]

COMPLETION_HEADER_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s*solution\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*solution(?:\s+\d+)?\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*answer\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*soln\.?\s*$", re.IGNORECASE),
]


def _first_present(row: dict[str, Any], candidates: tuple[str, ...], default: str) -> str:
    for key in candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _row_id(row: dict[str, Any], fallback_index: int) -> str:
    for key in ID_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _clean_problem_type(value: str) -> str:
    lowered = value.strip().lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[_/+-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def canonical_problem_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Other"

    cleaned = _clean_problem_type(value)
    for category, patterns in CATEGORY_PATTERNS:
        if any(pattern in cleaned for pattern in patterns):
            return category
    return "Other"


def infer_problem_type(row: dict[str, Any]) -> str:
    for key in PROBLEM_TYPE_KEYS:
        value = row.get(key)
        if value is not None:
            return canonical_problem_type(str(value))
    return "Other"


def _extract_source_final_answer(source: str, row: dict[str, Any], fallback: str) -> str:
    if source == "gsm8k":
        answer = row.get("answer")
        if isinstance(answer, str):
            match = GSM8K_ANSWER_RE.search(answer)
            if match:
                return match.group(1).strip()
    return fallback


def clean_completion_for_protocol(solution: str) -> str:
    if not solution.strip():
        return solution

    lines = solution.splitlines()
    start = 0
    while start < len(lines):
        line = lines[start].strip()
        if not line:
            start += 1
            continue
        if any(pattern.match(line) for pattern in COMPLETION_HEADER_PATTERNS):
            start += 1
            continue
        break
    cleaned = "\n".join(lines[start:]).strip()
    return cleaned or solution.strip()


def make_record(row: dict[str, Any], index: int, source: str, *, include_solution: bool) -> dict[str, str]:
    question = _first_present(row, QUESTION_KEYS, "")
    raw_solution = _first_present(row, SOLUTION_KEYS, "")
    final_answer = _first_present(row, FINAL_ANSWER_KEYS, "")
    final_answer = _extract_source_final_answer(source, row, final_answer)
    if not final_answer:
        final_answer = extract_relaxed_final_answer(raw_solution)
    if raw_solution:
        solution = ensure_boxed_final_answer(raw_solution, final_answer)
    elif final_answer:
        solution = f"Final answer: \\boxed{{{final_answer}}}"
    else:
        solution = ""
    final_answer = (
        extract_relaxed_final_answer(solution)
        or final_answer
        or extract_relaxed_final_answer(raw_solution)
        or ""
    )

    record = {
        "id": _row_id(row, index),
        "question": question,
        "final_answer": final_answer,
        "source": source,
    }
    if include_solution:
        record["solution"] = solution
    return record


def to_sft_record(row: dict[str, Any], index: int) -> dict[str, str]:
    record = make_record(row, index, "numinamath", include_solution=True)
    record["problem_type"] = infer_problem_type(row)
    return record


def to_eval_record(row: dict[str, Any], index: int, source: str) -> dict[str, str]:
    return make_record(row, index, source, include_solution=True)


def format_sft_text(question: str, solution: str) -> str:
    return f"Question:\n{question}\n\nSolution:\n{solution}"


def format_eval_text(question: str, final_answer: str) -> str:
    return format_sft_text(question, f"Final answer: \\boxed{{{final_answer}}}")


def format_protocol_prompt(question: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        "要求：\n"
        "请给出必要推理。\n"
        "最后一行必须严格写成：\n"
        "Final answer: \\boxed{...}\n\n"
        "Solution:\n"
    )
