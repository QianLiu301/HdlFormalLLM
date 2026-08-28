#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对配置里的模型在各服务商那边是否仍然可用

模型会被下线。Groq 就下线了 llama-3.3-70b-versatile —— 而它是 base01/base02
用的那个，307 次历史调用都基于它。请求一个已下线的模型返回 404，provider 捕获
异常后返回兜底文本，端点照样报 success=True，界面上看起来像「模型不会写
Verilog」。等到跑完一整批才发现，代价太大。

采集数据前跑一次：

    python benchmark/check_models_live.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests                                    # noqa: E402
import main as webapp                              # noqa: E402  （加载 config、设置环境变量）
from src.llm_providers import LLMFactory           # noqa: E402

# provider -> 列出模型的 URL。没有 /models 接口的（Gemini 用自己的 API 形态）
# 留空，改用一次极小的真实调用来判断。
MODELS_ENDPOINT = {
    'groq':     'https://api.groq.com/openai/v1/models',
    'openai':   'https://api.openai.com/v1/models',
    'deepseek': 'https://api.deepseek.com/models',
    'mistral':  'https://api.mistral.ai/v1/models',
    'together': 'https://api.together.xyz/v1/models',
    'qwen':     'https://chat-ai.academiccloud.de/v1/models',
    'llama':    'https://chat-ai.academiccloud.de/v1/models',
    'gptoss':   'https://chat-ai.academiccloud.de/v1/models',
    'glm':      'https://chat-ai.academiccloud.de/v1/models',
}


def list_models(provider, api_key):
    url = MODELS_ENDPOINT.get(provider)
    if not url:
        return None, 'no /models endpoint'
    try:
        r = requests.get(url, headers={'Authorization': f'Bearer {api_key}'},
                         proxies={'http': None, 'https': None}, timeout=30)
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:40]}'
    if r.status_code != 200:
        return None, f'HTTP {r.status_code}'
    try:
        body = r.json()
        # Together 直接返回一个裸 list，其余家包在 {"data": [...]} 里
        data = body if isinstance(body, list) else (
            body.get('data') or body.get('models') or [])
        return sorted(m['id'] for m in data if isinstance(m, dict) and 'id' in m), None
    except Exception as e:
        return None, f'parse: {str(e)[:40]}'


def probe(provider):
    """没有 /models 时，用一次最小调用判断模型是否可用。"""
    try:
        p = LLMFactory.create_provider(provider)
        call = getattr(p, '_call_api_text', None) or getattr(p, '_call_api', None)
        if call is None:
            return False, 'no _call_api (fell back to local template)'
        r = call('Reply with exactly: OK', max_tokens=200) or ''
        if 'Test ALU operation' in r or 'Given ALU operation' in r:
            return False, 'fallback text (API call failed)'
        return True, r[:24].replace('\n', ' ')
    except Exception as e:
        return False, f'{type(e).__name__}: {str(e)[:40]}'


def main():
    cfg = json.loads((ROOT / 'config' / 'llm_config.json').read_text(encoding='utf-8'))
    providers = cfg.get('providers', {})

    sys.path.insert(0, str(ROOT / 'benchmark'))
    import check_provider_lists as cpl
    active = sorted(cpl.factory_names())

    print(f"{'provider':<12}{'配置的模型':<34}{'状态'}")
    print('-' * 78)
    bad = []
    for name in active:
        model = (providers.get(name) or {}).get('model') or '(类默认值)'
        try:
            key = LLMFactory.create_provider(name).api_key
        except Exception as e:
            print(f"{name:<12}{model:<34}构造失败 {type(e).__name__}")
            bad.append(name)
            continue

        ids, err = list_models(name, key)
        if ids is not None:
            ok = model in ids
            note = '可用' if ok else f'✗ 不在服务商清单里（该处共 {len(ids)} 个模型）'
            if not ok:
                bad.append(name)
            print(f"{name:<12}{model:<34}{note}")
            if not ok:
                hint = [i for i in ids if not any(
                    k in i for k in ('whisper', 'guard', 'orpheus', 'embedding', 'tts'))]
                print(f"{'':<12}{'':<34}  该处可用: {', '.join(hint[:6])}"
                      + (' …' if len(hint) > 6 else ''))
        else:
            ok, detail = probe(name)
            if not ok:
                bad.append(name)
            print(f"{name:<12}{model:<34}{'可用（实调）' if ok else '✗ ' + detail}"
                  f"   [{err}]")

    print()
    if bad:
        print(f"有问题的 provider: {', '.join(bad)}")
        print("请求一个已下线的模型会返回 404，provider 随即降级成兜底文本，")
        print("而端点仍报 success=True —— 采集前务必先修掉。")
        return 1
    print("全部模型均可用。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
