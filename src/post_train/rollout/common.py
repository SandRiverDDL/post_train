from __future__ import annotations

from typing import Any


def shard_suffix(shard_index: int, num_shards: int) -> str:
    return f"shard{shard_index}-of-{num_shards}"


def select_shard_rows(rows: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError(f"num_shards 必须 >= 1，实际为 {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index 必须在 [0, {num_shards})，实际为 {shard_index}")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0}
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def render_chat_prompt(
    tokenizer: object,
    prompt: str,
    *,
    system_prompt: str | None = None,
    assistant_prefill: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("use_chat_template=true 但 tokenizer 没有 chat_template。")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    return str(rendered) + (assistant_prefill or "")
