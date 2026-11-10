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
    # 这个降级曾经完全无声：线上少装 PyYAML，模板全部加载失败，各 generator
    # 悄悄回退到内置 f-string，唯一症状是 Prompt 面板一片空白。生成内容虽然
    # 不变（模板本就是从那些 f-string 提取的），但版本切换与 override 全部失效
    # 而没有任何提示。至少让它在启动日志里喊一声。
    print("⚠️  PyYAML not installed — prompt templates in prompts/ cannot be "
          "loaded. Generators fall back to their built-in prompts; the Prompt "
          "panel will be empty and version/override selection will have no "
          "effect. Fix with: pip install PyYAML")

TEMPLATES_AVAILABLE = yaml is not None

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


def bdd_context_block(bdd_context: Optional[str]) -> str:
    """Specification-First 下追加到 DUV prompt 末尾的 BDD 规格段落。

    这段话此前在 4 个 generator 和 main.py 的流式端点里各存了一份，而
    generator 里那 4 份写在缩进的 f-string 中、每行平白多出 8 个空格——
    同一个功能会因为 Stream 开关不同而发出逐字不同的 prompt。对一个做
    受控对比的工具来说这是隐性变量，故收敛到此处一份。
    """
    if not bdd_context:
        return ""
    return f"""

IMPORTANT - BDD Specification Context (Specification-First Workflow):
The generated hardware MUST satisfy the following BDD test specifications.
Ensure all operations, flags, and behaviors described below are correctly implemented.

{bdd_context}

Make sure the module interface and behavior match the test expectations above.
"""


def user_request_block(text: Optional[str], for_scenarios: bool = False) -> str:
    """What the user typed in the Step 1 or Step 2 box, word for word.

    The box used to reach the model only through the values parsed from it (width,
    operations, opcodes, number of tests), so a line such as "Contrast SLT against
    SLTU" was dropped without a trace. This block comes before the Step 0 block,
    which wins where the two differ. Only the web endpoints set it.

    Scenarios must stay in the wording above: without that line a request such as
    "Concurrent read/write test" made the model invent steps that neither the
    rule library nor the testbench can read (14 to 17 per register-file run), and
    with new steps ruled out it wrote a same-cycle read as a write followed by a
    read, whose old value was then judged a scenario error (D41).

    Empty text gives an empty string, so the prompt stays as it was.
    """
    text = (text or '').strip()
    if not text:
        return ""
    block = ("\n\nThe user's request for this step, in their own words:\n\n"
             f"{text}\n\n"
             "Follow it together with the instructions above. Keep the module type, "
             "bit width, ports and output format given above.\n")
    if for_scenarios:
        block += ("Write every step in the exact wording shown above, with only the "
                  "columns shown above. Do not invent new steps; leave out anything "
                  "that cannot be written that way. Steps run one after another, so "
                  "also leave out anything that must happen in the same clock cycle.\n")
    return block


def decisions_block(record: Optional[dict]) -> str:
    """需求确认后追加到 DUV prompt 与 BDD prompt 末尾的段落。

    两个生成器拿到的是逐字相同的这一段：歧义在这里被用户一次定死，而不是
    交给两次独立的 LLM 调用各自猜——各自猜的结果在仿真时才以"期望值不符"
    的形式暴露，且看不出是哪一边猜错了。

    record 为空、或没有任何需求和决策时返回空串，prompt 与之前逐字相同。
    """
    if not record:
        return ""
    reqs = [r for r in (record.get('requirements') or []) if (r.get('text') or '').strip()]
    decs = [d for d in (record.get('decisions') or []) if (d.get('answer') or '').strip()]
    settled = record.get('settled') or []
    if not reqs and not decs and not settled:
        return ""
    lines = ["", "", "IMPORTANT - Confirmed Requirements and Decisions:",
             "The user confirmed the following. They apply to the design AND to its "
             "tests. Where they are more specific than the instructions above, "
             "follow them exactly."]
    if reqs:
        lines.append("")
        lines.append("Requirements:")
        lines += [f"- {r.get('id', '')}: {r['text'].strip()}" for r in reqs]
    if decs:
        lines.append("")
        lines.append("Decisions on points the requirements left open:")
        for d in decs:
            q = (d.get('question') or '').strip()
            lines.append(f"- {d.get('id', '')}: {q} -> {d['answer'].strip()}" if q
                         else f"- {d.get('id', '')}: {d['answer'].strip()}")
    if settled:
        lines.append("")
        lines.append("Fixed by the reference model the tests are judged against:")
        lines += [f"- {f['id']}: {f['fact']}" for f in settled]
    lines.append("")
    return "\n".join(lines)


_REPAIR_INTRO = {
    # failures of the last simulation, judged by the reference model
    'simulation': "A previous version of this design was simulated and made the errors "
                  "listed below. Write a corrected version: fix these errors, keep what "
                  "was correct, and keep the module name, the ports, their order and "
                  "their widths exactly as they are.",
    # the design alone does not compile
    'compile': "A previous version of this design does not compile with Icarus Verilog "
               "(iverilog -g2012). The compiler reported the errors listed below. Write a "
               "corrected version: fix every error without changing what the design does, "
               "and keep the module name, the ports, their order and their widths exactly "
               "as they are. Use Verilog-2001 only.",
    # the design compiles, but the testbench cannot connect to it
    'interface': "A previous version of this design compiles, but the testbench cannot "
                 "connect to it: its ports do not match the interface the testbench uses "
                 "(listed below). Write a corrected version whose module name and port names "
                 "are exactly the ones listed, keeping what the design does.",
}


def repair_feedback_block(feedback: Optional[str], previous: Optional[str] = None,
                          kind: str = 'simulation') -> str:
    """Appended to the design prompt when the design is regenerated with
    feedback: the failures of its last simulation (src/xdb/repair.design_feedback,
    kind 'simulation'; the reference model's verdicts, so a wrong scenario never
    turns into a design change), or the compiler's errors (kind 'compile', or
    'interface' when the testbench could not connect to the design).
    Empty feedback -> "".
    """
    if not (feedback or '').strip():
        return ""
    lines = ["", "", "IMPORTANT - Repair of a previous version:",
             _REPAIR_INTRO.get(kind, _REPAIR_INTRO['simulation']), "", feedback.strip()]
    if (previous or '').strip():
        lines += ["", "The previous version:", "```verilog", previous.strip(), "```"]
    lines += ["", "Return the complete corrected module.", ""]
    return "\n".join(lines)


def req_tag_instruction(record: Optional[dict]) -> str:
    """只给 BDD 生成器：让每个场景用 @req_N 标出它验证哪几条需求。

    标签必须是 @req_3 而不是 @REQ-3：解析器认的标签形式是 @\\w+，连字符会让
    整行标签失效，场景随之变成"无法映射"。
    """
    reqs = [r for r in ((record or {}).get('requirements') or [])
            if (r.get('text') or '').strip()]
    if not reqs:
        return ""
    return ("\nTRACEABILITY: put the IDs of the requirements each Scenario Outline "
            "verifies on that scenario's tag line, as extra tags of the form @req_1, "
            "@req_2 (REQ-1 -> @req_1). Keep the scenario's other tags. A scenario may "
            "list several requirements; use only IDs from the list above.\n")


def sha256(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
