from __future__ import annotations

import re
import random
from collections import defaultdict
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

BIG_MATH_ALLOWED_SOURCES = {"orca_math", "big_math", "math"}
BIG_MATH_EXCLUDED_SOURCES = {"aops_forum", "olympiads"}
BIG_MATH_ALLOWED_DOMAINS = {
    "applied mathematics",
    "algebra",
    "prealgebra",
    "simple equations",
    "equations and inequalities",
}
BIG_MATH_EXCLUDED_DOMAINS = {
    "combinatorics",
    "geometry",
    "number theory",
    "trigonometry",
    "complex numbers",
    "calculus",
}
BIG_MATH_PROOF_KEYWORDS = (
    "prove",
    "proof",
    "show that",
    "justify",
    "explain why",
    "derive",
    "discussion",
    "证明",
    "说明",
    "解释",
    "推导",
)
BIG_MATH_BANNED_ANSWER_MARKERS = ("%", r"\cup", r"\cap", r"\infty", r"^\circ", r"\{", r"\}")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
LATEX_DELIMITER_RE = re.compile(r"\\[\[\]\(\)]")
INTERVAL_ANSWER_RE = re.compile(r"^[\(\[].+,.+[\)\]]$")
INTEGER_RE = re.compile(r"^-?\d+$")
DECIMAL_RE = re.compile(r"^-?\d+\.\d+$")
SIMPLE_FRACTION_RE = re.compile(r"^-?\d+/\d+$")
SHORT_EXPRESSION_RE = re.compile(r"^[0-9A-Za-z+\-*/(). ]+$")


def scaled_category_targets(total_samples: int) -> dict[str, int]:
    base_total = sum(NUMINA_CATEGORY_TARGETS.values())
    if total_samples <= 0:
        raise ValueError("train_num_samples 必须大于 0")

    raw_targets = {
        category: total_samples * target / base_total
        for category, target in NUMINA_CATEGORY_TARGETS.items()
    }
    scaled = {category: int(value) for category, value in raw_targets.items()}
    remainder = total_samples - sum(scaled.values())
    ranking = sorted(
        raw_targets.items(),
        key=lambda item: (item[1] - scaled[item[0]], NUMINA_CATEGORY_TARGETS[item[0]]),
        reverse=True,
    )
    for category, _ in ranking[:remainder]:
        scaled[category] += 1
    return scaled


def sample_numinamath(rows: list[dict[str, Any]], rng: random.Random, total_samples: int) -> list[dict[str, Any]]:
    targets = scaled_category_targets(total_samples)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("problem_type", infer_problem_type(row)))].append(row)

    summary = {category: len(buckets.get(category, [])) for category in targets}
    missing = {
        category: target
        for category, target in targets.items()
        if len(buckets.get(category, [])) < target
    }
    if missing:
        raise ValueError(f"NuminaMath 分层抽样样本不足：{missing}；当前计数：{summary}")

    sampled: list[dict[str, Any]] = []
    for category, target in targets.items():
        bucket = list(buckets[category])
        rng.shuffle(bucket)
        sampled.extend(bucket[:target])

    rng.shuffle(sampled)
    return sampled


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


def format_protocol_prompt(question: str, *, prompt_version: str = "v1") -> str:
    if prompt_version == "v1":
        return (
            f"Question:\n{question}\n\n"
            "要求：\n"
            "请给出必要推理。\n"
            "最后一行必须严格写成：\n"
            "Final answer: \\boxed{...}\n\n"
            "Solution:\n"
        )
    if prompt_version == "v2":
        return (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"Question:\n{question}\n\n"
            "Solution:\n"
        )
    raise ValueError(f"不支持的 prompt_version: {prompt_version}")


def count_latex_markers(text: str) -> int:
    return len(LATEX_COMMAND_RE.findall(text)) + len(LATEX_DELIMITER_RE.findall(text))


def looks_like_proof_prompt(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in BIG_MATH_PROOF_KEYWORDS)


def is_allowed_short_answer(answer: str) -> bool:
    normalized = answer.strip()
    if not normalized:
        return False
    if any(marker in normalized for marker in BIG_MATH_BANNED_ANSWER_MARKERS):
        return False
    if INTERVAL_ANSWER_RE.match(normalized):
        return False
    if INTEGER_RE.match(normalized):
        return True
    if DECIMAL_RE.match(normalized):
        return True
    if SIMPLE_FRACTION_RE.match(normalized):
        return True
    if len(normalized) > 16:
        return False
    if "=" in normalized or "," in normalized or ";" in normalized:
        return False
    return bool(SHORT_EXPRESSION_RE.match(normalized))


def normalize_question_text(question: str) -> str:
    normalized = question.strip().lower()
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def domain_paths_text(domains: Any) -> str:
    if isinstance(domains, list):
        return " | ".join(str(domain) for domain in domains)
    return str(domains or "")


def has_allowed_big_math_domain(domains: Any) -> bool:
    lowered = domain_paths_text(domains).lower()
    return any(keyword in lowered for keyword in BIG_MATH_ALLOWED_DOMAINS)


def has_excluded_big_math_domain(domains: Any) -> bool:
    lowered = domain_paths_text(domains).lower()
    return any(keyword in lowered for keyword in BIG_MATH_EXCLUDED_DOMAINS)


def select_middle_by_solve_rate(rows: list[dict[str, Any]], *, rate_key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: float(row[rate_key]))
    lower_cut = int(len(ordered) * 0.2)
    upper_cut = len(ordered) - lower_cut
    return ordered[lower_cut:upper_cut]
