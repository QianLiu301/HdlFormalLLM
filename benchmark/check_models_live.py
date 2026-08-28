#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对配置与下拉框里的模型是否真的可用

判据是「实调能否返回结果」，不是「有没有列在 /models 里」——两者不等价：
Together 目录里的 Llama-3.1-405B、GWDG 目录里的 qwen3.8-27b 都在清单中，
实调却分别失败与 503。

模型也会被悄悄下线。Groq 下线了 llama-3.3-70b-versatile、DeepSeek 下线了
deepseek-chat，二者合计 641 次历史调用，base01 与 base02 都建立在它们之上。
请求已下线的模型返回 404，provider 捕获后返回兜底文本，端点仍报 success=True，
界面上看起来像「模型不会写 Verilog」。跑大批量采集之前先跑这个。

    python benchmark/check_models_live.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as webapp                              # noqa: E402  加载 config、设置环境变量
from src.llm_providers import LLMFactory           # noqa: E402

FALLBACK_MARKERS = ('Test ALU operation with various input values',
                    'Given ALU operation, When executed')
_END = "\n        };"


def dropdown_models():
    """前端 modelOptions 里列出的候选模型。"""
    html = (ROOT / 'static' / 'bdd_generator.html').read_text(encoding='utf-8')
    i = html.index('const modelOptions')
    j = html.index(_END, i)
    out = {}
    for m in re.finditer(r"'([\w\-]+)':\s*\[(.*?)\]", html[i:j], re.S):
        out[m.group(1)] = re.findall(r"value: '([^']+)'", m.group(2))
    return out


def probe(provider, model=None, attempts=3):
    """实调最小请求，失败重试。返回 (是否可用, 说明)。

    必须重试：学术云与部分厂商偶发 503 / 限流，单次失败会被误判成「模型不可用」。
    一个会误报的检查工具比没有更糟——用它的人会开始忽略它的输出。
    """
    import time
    detail = ''
    for i in range(attempts):
        ok, detail = _probe_once(provider, model)
        if ok:
            return True, detail
        if i < attempts - 1:
            time.sleep(3 * (i + 1))
    return False, detail + f'（重试 {attempts} 次均失败）'


def _probe_once(provider, model=None):
    try:
        p = (LLMFactory.create_provider(provider, model=model) if model
             else LLMFactory.create_provider(provider))
        call = getattr(p, '_call_api', None)
        if call is None:
            return False, 'fell back to local template (no api key?)'
        # 预算给足：gpt-oss 这类推理模型会先消耗一大段 reasoning
        r = call('Reply with exactly: OK', max_tokens=2000) or ''
        if any(m in r for m in FALLBACK_MARKERS):
            return False, 'fallback text — API call failed (see console for status)'
        return True, r.strip()[:20].replace("\n", ' ')
    except Exception as e:
        return False, f'{type(e).__name__}: {str(e)[:50]}'


def main():
    cfg = json.loads((ROOT / 'config' / 'llm_config.json').read_text(encoding='utf-8'))
    providers = cfg.get('providers', {})
    sys.path.insert(0, str(ROOT / 'benchmark'))
    import check_provider_lists as cpl
    active = sorted(cpl.factory_names())
    dd = dropdown_models()

    bad = []
    print(f"{'provider':<11}{'model':<44}{'来源':<8}状态")
    print('-' * 82)
    for name in active:
        configured = (providers.get(name) or {}).get('model')
        seen = []
        for model in [configured] + [m for m in dd.get(name, []) if m != configured]:
            if not model or model in seen:
                continue
            seen.append(model)
            src = '配置' if model == configured else '下拉框'
            ok, detail = probe(name, None if model == configured else model)
            if not ok:
                bad.append(f'{name}/{model}')
            print(f"{name:<11}{model:<44}{src:<8}{'可用' if ok else '✗ ' + detail}")

    print()
    if bad:
        print(f"不可用：{', '.join(bad)}")
        print("这些会静默降级成兜底文本而端点仍报成功——采集前先修掉，")
        print("从 config/llm_config.json 或 modelOptions 里移除。")
        return 1
    print("全部模型实调可用。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
