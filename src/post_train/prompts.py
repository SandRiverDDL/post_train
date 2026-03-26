from __future__ import annotations


def build_sft_prompt(question: str) -> str:
    return (
        "Solve the following math problem.\n"
        "Show your reasoning step by step.\n"
        "End your response with a single final answer in the format \\boxed{...}.\n\n"
        f"Problem:\n{question}"
    )


def build_eval_prompt(question: str) -> str:
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
