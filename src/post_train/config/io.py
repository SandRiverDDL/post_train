from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _load_omegaconf_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> DictConfig:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f"配置继承存在循环引用：{cycle}")

    cfg = OmegaConf.load(resolved_path)
    if cfg is None:
        cfg = OmegaConf.create({})
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"配置文件根节点必须是映射：{resolved_path}")

    extends_value = cfg.get("extends")
    child_payload = OmegaConf.to_container(cfg, resolve=False)
    child_payload.pop("extends", None)
    merged = OmegaConf.create({})

    if extends_value:
        if isinstance(extends_value, str):
            extends_items = [extends_value]
        elif isinstance(extends_value, list):
            extends_items = list(extends_value)
        else:
            raise ValueError(f"extends 必须是字符串或字符串列表：{resolved_path}")
        for extend_item in extends_items:
            parent_path = (resolved_path.parent / extend_item).resolve()
            parent_cfg = _load_omegaconf_with_extends(parent_path, (*stack, resolved_path))
            merged = OmegaConf.merge(merged, parent_cfg)

    return OmegaConf.merge(merged, OmegaConf.create(child_payload))


def load_yaml_config(path: str | Path) -> dict:
    cfg = _load_omegaconf_with_extends(Path(path))
    return OmegaConf.to_container(cfg, resolve=True)
