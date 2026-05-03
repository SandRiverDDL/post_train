from __future__ import annotations

from typing import Literal

PromptStyle = Literal["default", "justrl_math"]


def build_sft_prompt(question: str) -> str:
    return (
        "Solve the following math problem.\n"
        "Show your reasoning step by step.\n"
        "End your response with a single final answer in the format \\boxed{...}.\n\n"
        f"Problem:\n{question}"
    )


def build_eval_prompt(question: str) -> str:
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def build_math_prompt(question: str, *, style: PromptStyle) -> str:
    if style == "default":
        return build_sft_prompt(question)
    if style == "justrl_math":
        return build_eval_prompt(question)
    raise ValueError(f"未知 prompt_style：{style}")


def render_chat_prompt(
    tokenizer: object,
    prompt: str,
    *,
    system_prompt: str | None,
    assistant_prefill: str | None,
) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise ValueError("use_chat_template=true 但 tokenizer 没有 chat_template。")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
    return str(rendered) + (assistant_prefill or "")
