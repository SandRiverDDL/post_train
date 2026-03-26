from __future__ import annotations

import re
from typing import Optional

from math_verify import parse, verify

FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(
    r"(?im)^(?:final answer|answer)\s*:\s*(.+?)\s*$|^(?:the answer is|答案是|答案为)\s*[:：]?\s*(.+?)\s*$"
)
BOXED_FINAL_RE = re.compile(r"(?is)final answer\s*:\s*\\boxed\{")
BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed_content(text: str) -> Optional[str]:
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None

    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        index += 1
    return None


def extract_final_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed

    match = FINAL_ANSWER_RE.search(text)
    if match:
        answer = match.group(1).strip()
        first_line = answer.splitlines()[0].strip()
        return first_line or None
    return None


def extract_relaxed_final_answer(text: str) -> Optional[str]:
    strict = extract_final_answer(text)
    if strict:
        return strict

    matches = list(ANSWER_LINE_RE.finditer(text))
    if not matches:
        return None

    last_match = matches[-1]
    answer = (last_match.group(1) or last_match.group(2) or "").strip()
    if not answer:
        return None
    answer = answer.strip("$").strip()
    return extract_boxed_content(answer) or answer


def strip_answer_suffix(solution: str) -> str:
    lines = solution.rstrip().splitlines()
    while lines:
        candidate = lines[-1].strip()
        if not candidate:
            lines.pop()
            continue
        if ANSWER_LINE_RE.match(candidate):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def ensure_boxed_final_answer(solution: str, final_answer: str | None = None) -> str:
    answer = (final_answer or extract_relaxed_final_answer(solution) or "").strip()
    if not answer:
        return solution

    cleaned = strip_answer_suffix(solution)
    suffix = f"\\boxed{{{answer}}}"
    if not cleaned:
        return suffix
    return f"{cleaned}\n\n{suffix}"


def has_boxed_final_answer(text: str) -> bool:
    return bool(BOXED_RE.search(text))


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""

    normalized = answer.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.rstrip(".")
    normalized = normalized.replace("^", "**")

    if normalized.startswith("{") and normalized.endswith("}"):
        normalized = normalized[1:-1]
    return normalized


def parse_math_answer(text: str | None):
    if not text:
        return []
    return parse(text)


def are_equivalent(predicted: str | None, expected: str | None) -> bool:
    if not predicted or not expected:
        return False
    try:
        return bool(verify(parse_math_answer(expected), parse_math_answer(predicted)))
    except Exception:
        left = normalize_answer(predicted)
        right = normalize_answer(expected)
        if not left or not right:
            return False
        return left == right


def evaluate_prediction(
    prediction_text: str,
    expected_answer: str,
    *,
    require_boxed: bool,
) -> dict[str, object]:
    boxed = has_boxed_final_answer(prediction_text)
    parsed_answer = extract_final_answer(prediction_text) if require_boxed else extract_relaxed_final_answer(prediction_text)
    parse_ok = bool(parsed_answer)

    if not parsed_answer:
        parsed_answer = extract_relaxed_final_answer(prediction_text) if require_boxed else None

    correct = bool(parsed_answer) and are_equivalent(parsed_answer, expected_answer)
    if require_boxed and not boxed:
        correct = False

    return {
        "boxed": boxed,
        "parse_ok": parse_ok,
        "parsed_answer": parsed_answer or "",
        "correct": correct,
    }
