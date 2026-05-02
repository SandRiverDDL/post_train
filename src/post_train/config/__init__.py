from __future__ import annotations

from pathlib import Path

from .eval import EvalConfig, EvalTaskConfig
from .grpo import GRPODataConfig, GRPORewardConfig, GRPOTrainConfig
from .io import load_yaml_config
from .lightning_opd import LightningOPDDataConfig
from .on_policy import OnPolicyDataConfig, OnPolicyLoopConfig
from .stage1 import Stage1Math220kDataConfig, Stage1RSRCandidatesConfig, Stage1RSRSelectConfig
from .sft import SFTTrainConfig
from .simpo import SimPOConfig, SimPODataConfig
from .stage2 import Stage2DataConfig, Stage2HendrycksLongDataConfig, Stage2MixLongDataConfig
from .workflow import WorkflowConfig


def load_sft_config(path: str | Path) -> SFTTrainConfig:
    return SFTTrainConfig.model_validate(load_yaml_config(path))


def load_stage1_math220k_data_config(path: str | Path) -> Stage1Math220kDataConfig:
    return Stage1Math220kDataConfig.model_validate(load_yaml_config(path))


def load_stage1_rsr_candidates_config(path: str | Path) -> Stage1RSRCandidatesConfig:
    return Stage1RSRCandidatesConfig.model_validate(load_yaml_config(path))


def load_stage1_rsr_select_config(path: str | Path) -> Stage1RSRSelectConfig:
    return Stage1RSRSelectConfig.model_validate(load_yaml_config(path))


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(load_yaml_config(path))


def load_grpo_data_config(path: str | Path) -> GRPODataConfig:
    return GRPODataConfig.model_validate(load_yaml_config(path))


def load_grpo_reward_config(path: str | Path) -> GRPORewardConfig:
    return GRPORewardConfig.model_validate(load_yaml_config(path))


def load_grpo_train_config(path: str | Path) -> GRPOTrainConfig:
    return GRPOTrainConfig.model_validate(load_yaml_config(path))


def load_lightning_opd_data_config(path: str | Path) -> LightningOPDDataConfig:
    return LightningOPDDataConfig.model_validate(load_yaml_config(path))


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


def load_stage2_mix_long_data_config(path: str | Path) -> Stage2MixLongDataConfig:
    return Stage2MixLongDataConfig.model_validate(load_yaml_config(path))


def load_stage2_hendrycks_long_data_config(path: str | Path) -> Stage2HendrycksLongDataConfig:
    return Stage2HendrycksLongDataConfig.model_validate(load_yaml_config(path))


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    return WorkflowConfig.model_validate(load_yaml_config(path))


__all__ = [
    "EvalConfig",
    "EvalTaskConfig",
    "GRPODataConfig",
    "GRPORewardConfig",
    "GRPOTrainConfig",
    "LightningOPDDataConfig",
    "OnPolicyDataConfig",
    "OnPolicyLoopConfig",
    "SFTTrainConfig",
    "SimPOConfig",
    "SimPODataConfig",
    "Stage1Math220kDataConfig",
    "Stage1RSRCandidatesConfig",
    "Stage1RSRSelectConfig",
    "Stage2DataConfig",
    "Stage2HendrycksLongDataConfig",
    "Stage2MixLongDataConfig",
    "WorkflowConfig",
    "load_eval_config",
    "load_grpo_data_config",
    "load_grpo_reward_config",
    "load_grpo_train_config",
    "load_lightning_opd_data_config",
    "load_on_policy_data_config",
    "load_on_policy_loop_config",
    "load_sft_config",
    "load_stage1_math220k_data_config",
    "load_stage1_rsr_candidates_config",
    "load_stage1_rsr_select_config",
    "load_simpo_config",
    "load_simpo_data_config",
    "load_stage2_data_config",
    "load_stage2_hendrycks_long_data_config",
    "load_stage2_mix_long_data_config",
    "load_workflow_config",
    "load_yaml_config",
]
