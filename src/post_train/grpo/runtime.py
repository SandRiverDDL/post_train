from __future__ import annotations

from post_train.config import GRPOTrainConfig


GRPO_VLLM_SERVER_BASE_URL_ENV = "GRPO_VLLM_SERVER_BASE_URL"


def apply_grpo_runtime_env_overrides(
    cfg: GRPOTrainConfig,
    *,
    env: dict[str, str],
) -> GRPOTrainConfig:
    vllm_server_base_url = env.get(GRPO_VLLM_SERVER_BASE_URL_ENV)
    if vllm_server_base_url:
        # wrapper 会按实际占用端口启动 vLLM，trainer 必须使用同一个地址。
        return cfg.model_copy(update={"vllm_server_base_url": vllm_server_base_url})
    return cfg
