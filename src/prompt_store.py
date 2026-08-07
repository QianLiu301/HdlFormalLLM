"""
Prompt Store — prompt 模板的加载与渲染

把此前散落在各 generator 里、硬编码为 f-string 的 prompt 外置成
prompts/<stage>/<version>.yaml，使其可查看、可编辑、可版本化、可记录。
这是跨模型公平对比的前提：只有 prompt 文本本身可被外部核对，
"所有 provider 收到了相同 prompt" 这句话才是可验证的。

模板格式：
    version: v1
    stage: duv_alu
    system: |
      You are an expert Verilog hardware designer...
    user: |
      Generate a {{ bitwidth }}-bit ALU...
    variables: [bitwidth, module_name, ...]

占位符写作 {{ expr }}，其中 expr 是原 f-string 里的表达式原文
（例如 bitwidth、req['num_tests']、chr(10).join(...)）。渲染时按字符串
键精确匹配，不做求值——求值仍在各 generator 的 Python 代码里完成，
模板只负责文本。
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - 部署环境缺 PyYAML 时退回硬编码路径
    yaml = None

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PLACEHOLDER = re.compile(r"\{\{\s*(.*?)\s*\}\}", re.S)

# 模板文件读入后缓存；单次进程内 prompt 不会变
_cache: Dict[Tuple[str, str], dict] = {}


def available(stage: str):
    """列出某 stage 的可用版本，如 ['v1']。"""
    d = PROMPTS_DIR / stage
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def load(stage: str, version: str = "v1") -> Optional[dict]:
    """读取模板。缺文件或缺 PyYAML 时返回 None，调用方回退到内置 prompt。"""
    key = (stage, version)
    if key in _cache:
        return _cache[key]
    if yaml is None:
        return None
    path = PROMPTS_DIR / stage / f"{version}.yaml"
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  prompt template {stage}/{version} unreadable: {e}")
        return None
    _cache[key] = doc
    return doc


def render(text: str, variables: Dict[str, object]) -> str:
    """把 {{ expr }} 替换成 variables[expr]。

    未提供的占位符原样保留而不是抛错或替换成空串：宁可让缺失在输出里
    显形，也不要悄悄产出一份少了内容的 prompt。
    """
    def sub(m):
        name = m.group(1)
        if name in variables:
            return str(variables[name])
        return m.group(0)
    return _PLACEHOLDER.sub(sub, text)


def render_stage(stage: str, variables: Dict[str, object], version: str = "v1",
                 override: Optional[str] = None,
                 system_override: Optional[str] = None):
    """渲染某 stage 的模板，返回 (system, user)；模板不可用时返回 (None, None)。

    override / system_override 是用户在前端编辑后的文本；给出时代替文件内容，
    仍走同一套 {{ }} 替换，所以变量依然生效。二者可以只给其一。

    来源信息（版本、user/system 是否被覆盖）在此登记到 experiment_logger，
    随本次调用写入 llm_calls.extra——放在这里而不是各 generator 里，
    是为了不可能漏记。
    """
    doc = load(stage, version)
    if doc is None and override is None and system_override is None:
        return None, None

    user_overridden = override is not None
    sys_overridden = system_override is not None

    user_src = override if user_overridden else (doc or {}).get("user", "")
    system_src = system_override if sys_overridden else (doc or {}).get("system")

    _record(stage, version, user_overridden, sys_overridden)
    system = render(system_src, variables) if system_src else system_src
    return system, render(user_src, variables)


def _record(stage: str, version: str, user_overridden: bool,
            system_overridden: bool = False) -> None:
    try:
        try:
            from src.experiment_logger import record_call_meta
        except ImportError:
            from experiment_logger import record_call_meta
        record_call_meta(prompt_stage=stage,
                         prompt_template_version=version,
                         prompt_overridden=user_overridden,
                         system_prompt_overridden=system_overridden)
    except Exception:
        pass  # 记录失败不应影响生成


def sha256(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
