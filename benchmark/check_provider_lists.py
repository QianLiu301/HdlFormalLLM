#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider 名单一致性检查

同一批 provider 目前写在四个地方，彼此没有引用关系：

    src/llm_providers.py          LLMFactory 的注册表（唯一真正决定能不能用的）
    config/llm_config.json        key / model / endpoint
    static/bdd_generator.html     Step 1 与 Step 2 的下拉框
    static/experiment_dashboard.html   批量实验的 LLMS
    benchmark/run_baseline.py     ALL_PROVIDERS

这类重复已经出过三次问题：新加的 provider 网页能选、仪表盘跑不了；qwen 的
model 下拉框还留着换端点前的旧模型名；四个 generator 各自的名单不认识新
provider，未知名字静默回落到 Gemini。症状都是「看起来在用 A，实际用的是 B」。

    python benchmark/check_provider_lists.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# claude / grok 在界面里是注释掉的，local/mock 不是真的 provider
EXCLUDED = {'local', 'mock', 'claude', 'grok', 'anthropic', 'xai'}


def factory_names():
    """LLMFactory 注册表里的规范名（去掉别名：同一个类只留最短的那个名字）。"""
    src = (ROOT / 'src' / 'llm_providers.py').read_text(encoding='utf-8')
    block = src[src.index('providers = {', src.index('class LLMFactory')):]
    block = block[:block.index('}')]
    # 注释掉的注册项不算——停用某个 provider 的做法就是把那两行注释掉
    block = '\n'.join(l for l in block.split('\n') if not l.lstrip().startswith('#'))
    pairs = re.findall(r"'([\w\-]+)':\s*(\w+)", block)
    by_class = {}
    for name, cls in pairs:
        if name in EXCLUDED:
            continue
        by_class.setdefault(cls, []).append(name)
    # 取每个类首次注册的名字作为规范名：工厂里主名在前、别名在后。
    # 不能取最短的——那样会把 openai 判成 'gpt'、llama 判成 'meta'。
    return {v[0] for v in by_class.values()}


def config_names():
    cfg = json.loads((ROOT / 'config' / 'llm_config.json').read_text(encoding='utf-8'))
    return {k for k, v in cfg.get('providers', {}).items()
            if k not in EXCLUDED and isinstance(v, dict) and v.get('enabled', True)}


def html_select_names(select_id):
    html = (ROOT / 'static' / 'bdd_generator.html').read_text(encoding='utf-8')
    i = html.index(f'id="{select_id}"')
    j = html.index('</select>', i)
    body = html[i:j]
    # 注释掉的 option 不算
    body = re.sub(r'<!--.*?-->', '', body, flags=re.S)
    return set(re.findall(r'<option value="([\w\-]+)"', body)) - EXCLUDED


def dashboard_names():
    html = (ROOT / 'static' / 'experiment_dashboard.html').read_text(encoding='utf-8')
    i = html.index('const LLMS = [')
    j = html.index('];', i)
    return set(re.findall(r"'([\w\-]+)'", html[i:j])) - EXCLUDED


def baseline_names():
    src = (ROOT / 'benchmark' / 'run_baseline.py').read_text(encoding='utf-8')
    i = src.index('ALL_PROVIDERS = [')
    j = src.index(']', i)
    return set(re.findall(r"'([\w\-]+)'", src[i:j])) - EXCLUDED


def main():
    lists = {
        'LLMFactory 注册表': factory_names(),
        'config/llm_config.json': config_names(),
        'bdd_generator Step 1': html_select_names('hw-llm'),
        'bdd_generator Step 2': html_select_names('bdd-llm'),
        'experiment_dashboard': dashboard_names(),
        'run_baseline.ALL_PROVIDERS': baseline_names(),
    }
    reference = lists['LLMFactory 注册表']

    print(f"参照：LLMFactory 注册表 {len(reference)} 家")
    print(f"  {', '.join(sorted(reference))}\n")

    ok = True
    width = max(len(k) for k in lists)
    for name, names in lists.items():
        if name == 'LLMFactory 注册表':
            continue
        missing = reference - names
        extra = names - reference
        good = not missing and not extra
        ok &= good
        line = f"  [{'OK  ' if good else 'DIFF'}] {name:<{width}} {len(names)} 家"
        if missing:
            line += f"  缺少: {', '.join(sorted(missing))}"
        if extra:
            line += f"  多出: {', '.join(sorted(extra))}"
        print(line)

    print()
    if ok:
        print("全部一致。")
        return 0
    print("存在不一致：某处能选、另一处跑不了，或反过来。")
    return 1


if __name__ == '__main__':
    sys.exit(main())
