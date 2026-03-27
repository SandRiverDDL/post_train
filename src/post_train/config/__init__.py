from __future__ import annotations

from pathlib import Path

from .eval import EvalConfig, EvalTaskConfig
from .io import load_yaml_config
from .on_policy import OnPolicyDataConfig, OnPolicyLoopConfig
from .sft import SFTTrainConfig
from .simpo import SimPOConfig, SimPODataConfig
from .stage2 import Stage2DataConfig


def load_sft_config(path: str | Path) -> SFTTrainConfig:
    return SFTTrainConfig.model_validate(load_yaml_config(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(load_yaml_config(path))


def load_simpo_config(path: str | Path) -> SimPOConfig:
    return SimPOConfig.model_validate(load_yaml_config(path))


def load_simpo_data_config(path: str | Path) -> SimPODataConfig:
    return SimPODataConfig.model_validate(load_yaml_config(path))


def load_on_policy_data_config(path: str | Path) -> OnPolicyDataConfig:
    return OnPolicyDataConfig.model_validate(load_yaml_config(path))


def load_on_policy_loop_config(path: str | Path) -> OnPolicyLoopConfig:
    return OnPolicyLoopConfig.model_validate(load_yaml_config(path))


def load_stage2_data_config(path: str | Path) -> Stage2DataConfig:
    return Stage2DataConfig.model_validate(load_yaml_config(path))


__all__ = [
    "EvalConfig",
    "EvalTaskConfig",
    "OnPolicyDataConfig",
    "OnPolicyLoopConfig",
    "SFTTrainConfig",
    "SimPOConfig",
    "SimPODataConfig",
    "Stage2DataConfig",
    "load_eval_config",
    "load_on_policy_data_config",
    "load_on_policy_loop_config",
    "load_sft_config",
    "load_simpo_config",
    "load_simpo_data_config",
    "load_stage2_data_config",
    "load_yaml_config",
]
