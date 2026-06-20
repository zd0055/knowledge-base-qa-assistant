"""配置加载模块"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def get_project_root() -> Path:
    """返回项目根目录"""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """加载 YAML 配置，合并环境变量"""
    load_dotenv()

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 从环境变量注入敏感值
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        config.setdefault("llm", {}).setdefault("deepseek", {})["api_key"] = deepseek_key

    ollama_base = os.getenv("OLLAMA_BASE_URL")
    if ollama_base:
        config.setdefault("llm", {}).setdefault("ollama", {})["base_url"] = ollama_base

    # 将相对路径解析为绝对路径
    root = get_project_root()
    vs = config.get("vector_store", {})
    if vs.get("persist_directory") and not Path(vs["persist_directory"]).is_absolute():
        vs["persist_directory"] = str(root / vs["persist_directory"])
    docs_cfg = config.get("documents", {})
    if docs_cfg.get("directory") and not Path(docs_cfg["directory"]).is_absolute():
        docs_cfg["directory"] = str(root / docs_cfg["directory"])

    return config


def resolve_path(config: Dict[str, Any], key: str, default: str = "") -> str:
    """将配置中的相对路径解析为绝对路径"""
    rel_path = config
    for k in key.split("."):
        if isinstance(rel_path, dict):
            rel_path = rel_path.get(k, {})
        else:
            return default
    if not isinstance(rel_path, str):
        return default
    path = Path(rel_path)
    if not path.is_absolute():
        path = get_project_root() / path
    return str(path)
