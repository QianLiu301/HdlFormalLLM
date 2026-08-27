"""
LLM Provider Module
Supports multiple LLM APIs including FREE options for generating BDD scenario descriptions
Updated to support GPT-5 series models: gpt-5.1-codex, gpt-5.1, gpt-5, gpt-5-mini

Fixed: gpt-5.1-codex uses completions endpoint (v1/completions) not chat completions
Other GPT-5 models use chat completions endpoint (v1/chat/completions)

🔧 FIXED: _fallback_description now returns JSON format for intent parsing compatibility
"""

import os
import json
import time
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests

# 尝试导入 google-genai,如果失败则标记
try:
    from google import genai
    from google.genai import errors

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("⚠️  google-genai not installed. Install with: pip install -U google-genai")

# Gemini 走 SDK 还是 REST。历史上写作 `if GENAI_AVAILABLE and False:`，
# 即使 SDK 已安装也强制走 REST；该写法自仓库首次提交(887326e, 2025-12-24)即存在，
# 无 commit 记录说明原因。改为显式常量以消除 `and False`，行为保持不变（仍为 REST）。
#
# 重新启用前必须先修复 GeminiProvider.__init__ 中 SDK 分支的三个已知缺陷：
#   1. 只设置 self.models_to_try，不设置 self.model
#      -> experiment_logger 的 getattr(self,'model') 取到 None，SDK 调用的
#         model 字段会全部记成 NULL，实验数据不可追溯
#   2. 构造函数的 model 参数被静默丢弃（REST 分支用 `model or "gemini-2.5-flash"`）
#      -> 调用方指定的模型不生效
#   3. models_to_try 里是已退役的模型名（gemini-2.0-flash-exp / gemini-1.5-*），
#      而 REST 分支用的是 gemini-2.5-flash
USE_GEMINI_SDK = False

# 尝试导入 openai SDK,如果失败则标记
try:
    from openai import OpenAI

    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    print("⚠️  openai SDK not installed. Install with: pip install openai")

# 实验记录系统：自动记录所有 LLM 调用到 SQLite（可选依赖，失败不影响运行）
try:
    from src.experiment_logger import (instrument_class as _exp_instrument,
                                       record_call_meta as _record_meta)
except ImportError:
    try:
        from experiment_logger import (instrument_class as _exp_instrument,
                                       record_call_meta as _record_meta)
    except ImportError:
        _exp_instrument = None

        def _record_meta(**fields):  # logger 不可用时的空实现
            pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    # ------------------------------------------------------------------
    # 采样参数
    #
    # DEFAULT_SAMPLING 里填的必须是该 provider 当前**硬编码的真实值**，
    # 按 api_path 分组（同一 provider 的不同实现路径取值可能不同，例如
    # Gemini 的 rest=0.4 / stream=0.3）。这些值刻意不做统一——不一致是既有
    # 事实，本机制只是把它变得可见、可覆盖、可记录。
    #
    # 调用方不传 sampling 时，_merge_sampling 原样返回这些默认值，
    # 因此 payload 与引入本机制之前逐字节一致。
    # ------------------------------------------------------------------
    DEFAULT_SAMPLING: Dict[str, Dict[str, Any]] = {}

    # 该 provider 真正支持的逻辑参数名。未列出的参数会被跳过并记为 unsupported，
    # 而不是塞进 payload 让 API 报错或静默忽略。
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'max_tokens'}

    # 逻辑名 -> API 字段名，仅在两者不同时才需要列出（如 Mistral 的 random_seed）
    SAMPLING_FIELD_MAP: Dict[str, str] = {}

    # 本工具允许用户输入的区间——注意这是**策略**，不等于各家 API 的合法区间。
    # 和 SUPPORTED_SAMPLING 一样由 provider 类声明、经 /api/sampling-info 下发；
    # 在 JS 里另抄一份会与后端漂移。
    #
    # temperature 上界取 1 而非部分 API 允许的 2：>1 时输出退化很快，对生成
    # Verilog 没有实用价值，而 1 是所有 provider 都接受的值（Anthropic 的硬
    # 上界恰好就是 1）。统一成 1 也让跨模型对比的取值范围一致。
    SAMPLING_RANGES: Dict[str, Dict[str, Any]] = {
        'temperature': {'min': 0, 'max': 1, 'step': 0.05},
        'top_p':       {'min': 0, 'max': 1, 'step': 0.05},
        # seed 只是标签，取值大小无意义；限成 32 位正整数以挡住手滑输入的天文数字
        'seed':        {'min': 0, 'max': 2147483647, 'step': 1},
        # max_tokens 的真实上界随模型而变，这里只挡明显不合理的输入
        'max_tokens':  {'min': 1, 'max': 32768, 'step': 1},
    }

    def __init_subclass__(cls, **kwargs):
        """子类定义时自动包装其 _call_api* 方法，实现调用级实验记录"""
        super().__init_subclass__(**kwargs)
        if _exp_instrument is not None:
            _exp_instrument(cls)

    def _forced_sampling(self, api_path: str = 'default') -> Dict[str, Any]:
        """返回不可被用户覆盖的参数。默认没有；OpenAI 的 GPT-5 系列覆盖此方法
        （其 API 只接受 temperature=1）。"""
        return {}

    def _merge_sampling(self, api_path: str = 'default', override: Optional[Dict] = None):
        """合并 provider 默认值与用户覆盖，过滤不支持的参数。

        返回 (payload_params, source, ignored)：
          payload_params  已映射成该 provider API 字段名、可直接并入 payload
          source          {逻辑参数名: 'user' | 'default' | 'forced' |
                                       'unsupported' | 'api_default'}
          ignored         被丢弃的参数名列表（不支持，或被强制值覆盖）

        max_tokens 不在此处理：它本就是 _call_api 的形参，各 provider 的字段名
        也不同（max_tokens / max_completion_tokens / maxOutputTokens），
        由调用点用 _resolve_max_tokens 单独解析。
        """
        defaults = dict(self.DEFAULT_SAMPLING.get(api_path, {}))
        forced = self._forced_sampling(api_path)
        override = {k: v for k, v in (override or {}).items() if v is not None}

        params, source, ignored = {}, {}, []

        for name in ('temperature', 'top_p', 'seed'):
            if name == 'max_tokens':
                continue
            user_val = override.get(name)

            if name in forced:
                params[name] = forced[name]
                source[name] = 'forced'
                if user_val is not None and user_val != forced[name]:
                    ignored.append(name)
                continue

            if name not in self.SUPPORTED_SAMPLING:
                # 无论用户是否传值都标注，让 sampling_source 完整反映 provider 能力；
                # 只有用户确实传了值才算“被忽略”
                source[name] = 'unsupported'
                if user_val is not None:
                    ignored.append(name)
                continue

            if user_val is not None:
                params[name] = user_val
                source[name] = 'user'
            elif name in defaults:
                params[name] = defaults[name]
                source[name] = 'default'
            else:
                # provider 无默认值且用户没填 -> 根本不发送该字段，
                # 让 API 自己的默认值生效（Claude 的 temperature 即如此）
                source[name] = 'api_default'

        # max_tokens 之外的 defaults 若未被上面覆盖，也要带上（防止漏发）
        for name, val in defaults.items():
            if name not in params and name not in source:
                params[name] = val
                source[name] = 'default'

        # 登记本次实际生效的采样事实，供埋点写入 llm_calls.extra。
        # 放在这里而不是各 provider 里：所有 provider 都会经过本方法，
        # 因此无需逐个添加登记代码，也不会漏。
        _record_meta(sampling_source=source,
                     sampling_ignored=ignored,
                     **{k: v for k, v in params.items()})

        # 映射成 API 字段名
        mapped = {self.SAMPLING_FIELD_MAP.get(k, k): v for k, v in params.items()}
        return mapped, source, ignored

    def _resolve_max_tokens(self, max_tokens: int, override: Optional[Dict] = None):
        """用户覆盖优先于调用方传入的 max_tokens。返回 (值, 来源)。"""
        user_val = (override or {}).get('max_tokens')
        value, src = (int(user_val), 'user') if user_val else (max_tokens, 'default')
        _record_meta(max_tokens=value, max_tokens_source=src)
        return value, src

    def _record_response_meta(self, result) -> None:
        """从 API 响应里提取 finish_reason / system_fingerprint 并登记。

        各家字段名不同，这里统一探测，避免在每个 provider 里写一遍：
          OpenAI 兼容  choices[0].finish_reason + system_fingerprint
          Gemini       candidates[0].finishReason
          Claude       stop_reason
        """
        if not isinstance(result, dict):
            # OpenAI SDK 返回的是对象而非 dict：按属性取
            choices = getattr(result, 'choices', None)
            if choices:
                _record_meta(finish_reason=getattr(choices[0], 'finish_reason', None),
                             system_fingerprint=getattr(result, 'system_fingerprint', None))
            return
        finish = None
        choices = result.get('choices')
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish = choices[0].get('finish_reason')
        if finish is None:
            cands = result.get('candidates')
            if isinstance(cands, list) and cands and isinstance(cands[0], dict):
                finish = cands[0].get('finishReason')
        if finish is None:
            finish = result.get('stop_reason')
        _record_meta(finish_reason=finish,
                     system_fingerprint=result.get('system_fingerprint'))


    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """
        获取代理配置（从环境变量）

        如果环境变量中设置了代理，返回代理字典
        否则返回None（不使用代理）

        由benchmark_runner.py的_setup_proxy()设置环境变量
        """
        proxies = None

        if os.environ.get('HTTPS_PROXY'):
            proxies = {
                'http': os.environ.get('HTTP_PROXY', ''),
                'https': os.environ.get('HTTPS_PROXY', '')
            }
            # 只在第一次使用时打印（避免重复日志）
            if not hasattr(self, '_proxy_logged'):
                print(f"  🌐 Using proxy: {proxies['https']}")
                self._proxy_logged = True

        return proxies

    @abstractmethod
    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        pass

    @abstractmethod
    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        pass

    def _parse_intent_from_prompt(self, prompt: str) -> Dict:
        """
        🔧 从 prompt 中解析 intent 信息用于 fallback

        当 API 失败时，尝试从原始 prompt 中提取关键信息
        """
        text = prompt.lower()

        # 检测操作类型
        operation = "ADD"
        for op in ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "SHL", "SHR"]:
            if op.lower() in text:
                operation = op
                break

        # 检测条件
        condition = "random"
        if "equal" in text or "same" in text or "a = b" in text or "a=b" in text:
            condition = "A = B"
        elif "greater" in text or "a > b" in text or "a>b" in text:
            condition = "A > B"
        elif "less" in text or "a < b" in text or "a<b" in text:
            condition = "A < B"
        elif "random" in text or "various" in text:
            condition = "random"

        # 检测示例数量
        num_match = re.search(r"(\d+)\s*(?:examples?|cases?|scenarios?)", text)
        num_examples = int(num_match.group(1)) if num_match else 3

        return {
            "operation": operation,
            "condition": condition,
            "scenario_name": f"{operation} with {condition}",
            "num_examples": num_examples,
            "tags": ["arithmetic"],
            "_fallback": True,
            "_fallback_reason": "API call failed, using local parsing"
        }

    def _fallback_intent_json(self, prompt: str) -> str:
        """
        🔧 返回 JSON 格式的 fallback intent

        这个方法专门用于 _understand_intent() 调用失败时返回有效的 JSON
        """
        intent = self._parse_intent_from_prompt(prompt)
        return json.dumps(intent, ensure_ascii=False, indent=2)


# ========== FREE PROVIDERS ==========


class GeminiProvider(LLMProvider):
    # rest 与 stream 的硬编码温度本就不同（0.4 / 0.3），如实记录、不做统一。
    # sdk 为死代码（见 USE_GEMINI_SDK），不填。采样参数位于 generationConfig 内。
    DEFAULT_SAMPLING = {'rest': {'temperature': 0.4},
                        'stream': {'temperature': 0.3}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}
    SAMPLING_FIELD_MAP = {'top_p': 'topP'}

    """
    Google Gemini API Provider - FREE!

    Free tier: 60 requests per minute
    How to get API key:
    1. Visit: https://makersuite.google.com/app/apikey
    2. Click "Create API Key"
    3. Copy the key

    Note: Requires Google account
    Now uses google-genai SDK with retry and model fallback
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Get free key at: https://makersuite.google.com/app/apikey")

        # 如果 google-genai 可用且 SDK 路径已启用,使用新的 SDK（见 USE_GEMINI_SDK 说明）
        if GENAI_AVAILABLE and USE_GEMINI_SDK:
            # 使用新的模型名称和 SDK
            self.models_to_try = [ "gemini-2.0-flash-exp",  # Best for code
                                   "gemini-1.5-pro-latest",
                                   "gemini-1.5-flash-latest"]
            self.client = genai.Client(api_key=self.api_key)
            self.use_sdk = True
            self.max_retries = 3
            self.sleep_seconds = 2.0
        else:
            # 降级到旧的 REST API 方式
            self.model = model or "gemini-2.5-flash"
            self.use_sdk = False
            print(f"🔷 [Gemini] Initialized with model: {self.model}")
            print("⚠️  Using REST API fallback. For better reliability, install: pip install -U google-genai")

        # 🔧 保存最近的 prompt 用于 fallback 解析
        self._last_prompt = ""

    def _call_api_sdk(self, prompt: str, max_tokens: int = 8192, system_prompt: str = None) -> str:
        """使用新的 google-genai SDK 调用 API,包含重试和模型降级"""
        self._last_prompt = prompt  # 🔧 保存 prompt
        last_error = None

        for model_name in self.models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config={                          # ← 添加这个
                            "max_output_tokens": max_tokens,
                            "temperature": 0.7,
                        }
                    )
                    return response.text

                except errors.ServerError as e:
                    # 处理 5xx 错误(包括 503 overloaded)
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(self.sleep_seconds)
                    else:
                        break  # 这个模型的重试次数用完了

                except Exception as e:
                    # 其他错误,不需要重试
                    last_error = e
                    break

            # 如果不是最后一个模型,继续尝试下一个
            if model_name != self.models_to_try[-1]:
                continue

        # 所有尝试失败,返回 fallback
        print(f"⚠️  Gemini API failed after all attempts: {last_error}")
        return self._fallback_description(prompt)

    def _call_api_rest(self, prompt: str, max_tokens: int = 8192, system_prompt: str = None,
                       sampling: Optional[Dict] = None) -> str:
        # 修正模型名称 (改为稳定版)
        model_name = self.model if "gemini" in self.model else "gemini-2.0-flash-exp"
        max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
        print(f"🔷 [Gemini] Calling REST API - Model: {model_name}, max_tokens: {max_tokens}")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

        # 采样参数进 generationConfig；默认 temperature=0.4（代码生成用低随机性）
        gen_config = {"maxOutputTokens": max_tokens}
        gen_config.update(self._merge_sampling('rest', sampling)[0])
        gen_config["stopSequences"] = []

        # 正确构建 payload，加入 system_instruction
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }

        # 如果有 system_prompt，按照 Gemini API 规范添加
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

        try:
            proxies = self._get_proxies()
            response = requests.post(url, json=payload, timeout=60, proxies=proxies)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)

            # 增加安全性检查，防止 index error
            if 'candidates' in result and result['candidates'][0]['content']['parts']:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            return "Error: No content generated."
        except Exception as e:
            print(f"⚠️  Gemini REST API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 8192, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        """
        Streaming API call for Gemini
        Uses streamGenerateContent endpoint with alt=sse for proper SSE format
        """
        # 使用 alt=sse 获取标准 SSE 格式，避免 JSON 数组解析问题
        max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
        print(f"🔷 [Gemini] Calling Stream API - Model: {self.model}, max_tokens: {max_tokens}")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}"
            f":streamGenerateContent?alt=sse&key={self.api_key}"
        )
 
        gen_config = {"maxOutputTokens": max_tokens}
        gen_config.update(self._merge_sampling('stream', sampling)[0])

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }
 
        if system_prompt:
            payload["system_instruction"] = {"parts": [{"text": system_prompt}]}
 
        try:
            proxies = self._get_proxies()
            response = requests.post(
                url,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=300  # 增大超时
            )
            response.raise_for_status()
 
            # 使用 SSE 格式解析 (alt=sse)，每行以 "data: " 开头
            buffer = ""
            for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
 
                # 按行处理 SSE events
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
 
                    if not line or not line.startswith('data: '):
                        continue
 
                    json_str = line[6:]  # 去掉 "data: " 前缀
                    if not json_str:
                        continue
 
                    try:
                        data = json.loads(json_str)
                        if 'candidates' in data:
                            self._record_response_meta(data)   # 末个 chunk 带 finishReason
                            for candidate in data['candidates']:
                                if 'content' in candidate:
                                    for part in candidate['content'].get('parts', []):
                                        if 'text' in part:
                                            yield part['text']
                    except json.JSONDecodeError:
                        continue
 
        except Exception as e:
            print(f"⚠️ Gemini streaming failed: {e}, falling back to REST API")
            result = self._call_api_rest(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def _call_api(self, prompt: str, max_tokens: int = 8192, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        """统一的 API 调用接口"""
        if self.use_sdk:
            return self._call_api_sdk(prompt, max_tokens, system_prompt)
        else:
            return self._call_api_rest(prompt, max_tokens, system_prompt, sampling=sampling)

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """
        🔧 修复: Fallback 返回 JSON 格式以支持 intent 解析

        当 API 失败时，分析 prompt 并返回 JSON 格式的 intent
        """
        # 检查是否是 intent 解析请求（通过检查 prompt 中是否包含 JSON 相关指令）
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)

        # 否则返回普通文本描述
        return "Test ALU operation with various input values and verify correct output"


class GroqProvider(LLMProvider):
    """
    Groq API Provider - FREE and FAST!

    Free tier: 30 requests per minute, 14,400 per day
    How to get API key:
    1. Visit: https://console.groq.com/keys
    2. Sign up (free, no credit card needed)
    3. Create API key

    Models: llama3-70b, mixtral-8x7b, gemma-7b
    """

    # OpenAI 兼容 API：seed 有文档支持
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Groq API key not provided. Get free key at: https://console.groq.com/keys")

    def _call_api(self, prompt: str, max_tokens: int = 200, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        """Call Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
        params, _, _ = self._merge_sampling('default', sampling)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
        }
        payload.update(params)

        try:
            # 🆕 添加代理支持
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30, proxies=proxies)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Groq API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        """
        Streaming API call for Groq
        Yields chunks of generated text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            proxies = self._get_proxies()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"⚠️ Groq streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """
        🔧 修复: Fallback 返回 JSON 格式以支持 intent 解析
        """
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class DeepSeekProvider(LLMProvider):
    # seed 未见于 DeepSeek 官方文档，保守起见不声明支持
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'max_tokens'}

    """
    DeepSeek API Provider - FREE!

    Free tier available
    How to get API key:
    1. Visit: https://platform.deepseek.com/
    2. Sign up
    3. Get API key

    Model: deepseek-chat
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Get free key at: https://platform.deepseek.com/")

    def _get_proxies(self) -> Dict[str, None]:
        """
        DeepSeek 不需要代理（国内 API）。
        注意: requests 里 proxies=None 表示"使用环境变量代理"，
        必须显式传 {'http': None, 'https': None} 才能真正直连。
        """
        return {'http': None, 'https': None}
    def _call_api(self, prompt: str, max_tokens: int = 200, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        """
        Call DeepSeek API with detailed debug output

        Enhanced with comprehensive debugging similar to OpenAI provider
        """
        # ============================================================
        # 调试信息：调用参数
        # ============================================================
        print(f"   🔍 [DEBUG][DeepSeek._call_api] Called with max_tokens={max_tokens}")
        print(f"   🔍 [DEBUG] Model: {self.model}")
        print(f"   🔍 [DEBUG] Prompt length: {len(prompt)} chars")
        if system_prompt:
            print(f"   🔍 [DEBUG] System prompt length: {len(system_prompt)} chars")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('default', sampling)[0])

        # ============================================================
        # 调试信息：请求详情
        # ============================================================
        print(f"   🔍 [DEBUG] Request URL: {self.api_url}")
        print(f"   🔍 [DEBUG] Request payload: model={self.model}, messages=2, max_tokens={max_tokens}, temp=0.7")

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # ============================================================
                # 调试信息：重试次数（仅当 attempt > 0）
                # ============================================================
                if attempt > 0:
                    print(f"   🔄 Retrying...")
                    print(f"   🔄 [DEBUG] Retry {attempt}/{max_retries} - max_tokens: {max_tokens}")

                # ============================================================
                # 调试信息：正在发送请求
                # ============================================================
                print(f"   📡 [DEBUG] Sending request to DeepSeek API... (attempt {attempt + 1}/{max_retries})")

                # 🆕 添加代理支持
                proxies = self._get_proxies()
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60, proxies=proxies)

                # ============================================================
                # 调试信息：响应状态
                # ============================================================
                print(f"   📥 [DEBUG] Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    self._record_response_meta(result)
                    content = result['choices'][0]['message']['content'].strip()

                    # ============================================================
                    # 调试信息：成功和 token 统计
                    # ============================================================
                    usage = result.get('usage', {})
                    print(f"   ✅ [DEBUG] API call successful")
                    print(f"   📊 [DEBUG] Response length: {len(content)} chars")

                    if usage:
                        prompt_tokens = usage.get('prompt_tokens', 'N/A')
                        completion_tokens = usage.get('completion_tokens', 'N/A')
                        total_tokens = usage.get('total_tokens', 'N/A')
                        print(f"   📊 [DEBUG] Token usage: prompt={prompt_tokens}, "
                              f"completion={completion_tokens}, total={total_tokens}")

                    # 显示完成原因
                    finish_reason = result['choices'][0].get('finish_reason', 'unknown')
                    print(f"   🎯 [DEBUG] Finish reason: {finish_reason}")

                    # 显示模型信息（如果有）
                    if 'model' in result:
                        print(f"   🤖 [DEBUG] Model used: {result['model']}")

                    print(f"   ✅ [DEBUG][DeepSeek._call_api] Returning response ({len(content)} chars)")
                    return content
                else:
                    # ============================================================
                    # 调试信息：详细的错误信息
                    # ============================================================
                    error_detail = response.text
                    print(f"   ❌ [ERROR] API request failed: Status {response.status_code}")
                    print(f"   ❌ [ERROR] Error detail: {error_detail[:200]}")  # 只显示前200字符

                    if attempt < max_retries - 1:
                        print(f"   ⏳ [DEBUG] Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        print(f"   ❌ [ERROR] All {max_retries} attempts failed")
                        print(f"   ⚠️  [WARN] Returning fallback response")
                        return self._fallback_description(prompt)

            except requests.exceptions.Timeout:
                print(f"   ❌ [ERROR] Request timeout (60s)")
                if attempt < max_retries - 1:
                    print(f"   🔄 [DEBUG] Retrying after timeout...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"   ⚠️  [WARN] Timeout after all retries, returning fallback")
                    return self._fallback_description(prompt)

            except requests.exceptions.RequestException as e:
                print(f"   ❌ [ERROR] Network error: {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"   🔄 [DEBUG] Retrying after network error...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"   ⚠️  [WARN] Network error after all retries, returning fallback")
                    return self._fallback_description(prompt)

            except Exception as e:
                print(f"   ❌ [ERROR] Unexpected error: {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"   🔄 [DEBUG] Retrying after unexpected error...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"   ⚠️  [WARN] Unexpected error after all retries")
                    print(f"   ⚠️  DeepSeek API request failed: {e}")
                    return self._fallback_description(prompt)

        # 应该不会到达这里
        print(f"   ⚠️  [WARN] Max retries exceeded, returning fallback")
        return self._fallback_description(prompt)

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """
        🔧 修复: Fallback 返回 JSON 格式以支持 intent 解析
        """
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        """
        Streaming API call for DeepSeek
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            proxies = self._get_proxies()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=180
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"⚠️ DeepSeek streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result


# ========== PAID PROVIDERS ==========


class OpenAIProvider(LLMProvider):
    # GPT-5 系列的 API 只接受 temperature=1，见 _forced_sampling；
    # 旧模型沿用此前硬编码的 0.7。
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'json_mode': {'temperature': 0.7},
                        'text': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}

    def _forced_sampling(self, api_path: str = 'default') -> Dict[str, Any]:
        # GPT-5 系列拒绝 temperature != 1，用户传入的值只能记为 ignored
        if self._is_gpt5_model(self.model):
            return {'temperature': 1}
        return {}

    """
    OpenAI API Provider - PAID

    Supports GPT-5 series models including gpt-5.1-codex

    🔧 API Endpoint Usage:
    - Codex models (gpt-5.1-codex) → completions endpoint (v1/completions)
    - Other GPT-5 models (gpt-5, gpt-5-mini, gpt-5.1) → chat completions (v1/chat/completions)

    Note: Codex models return plain text; other models use JSON mode
    """

    GPT5_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.1-codex"]

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini"):
        if not OPENAI_SDK_AVAILABLE:
            raise ImportError("OpenAI SDK required. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.max_retries = 3

    def _is_gpt5_model(self, model: str) -> bool:
        """检查是否为 GPT-5 系列模型"""
        return model in self.GPT5_MODELS

    def _is_codex_model(self, model: str) -> bool:
        """
        检查是否为 Codex 模型 (使用 completions 接口)

        Codex 模型（包括 gpt-5.1-codex）使用 completions 端点，不是 chat completions
        """
        return 'codex' in model.lower()

    def _call_api_completions(
            self,
            prompt: str,
            max_tokens: int = 500,
            retry_count: int = 0
    ) -> str:
        """
        使用 OpenAI Completions API (用于 codex 模型)

        注意: Codex 模型不支持 JSON mode, 返回纯文本
        """
        if retry_count >= self.max_retries:
            print(f"❌ [ERROR] Max retries ({self.max_retries}) reached")
            return self._fallback_text()

        try:
            if retry_count == 0:
                print(f"🔍 [DEBUG] Completions Request - Model: {self.model}, max_tokens: {max_tokens}")
            else:
                print(f"🔄 [DEBUG] Retry {retry_count}/{self.max_retries} - max_tokens: {max_tokens}")

            # 调用 completions API
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=1.0
            )

            self._record_response_meta(response)

            # 调试信息
            print(f"🔗 [DEBUG] Response ID: {response.id}")
            print(f"🤖 [DEBUG] Model: {response.model}")
            print(f"📊 [DEBUG] finish_reason: {response.choices[0].finish_reason}")

            # 打印 token 使用情况
            if hasattr(response, 'usage') and response.usage:
                print(f"💰 [DEBUG] Tokens - "
                      f"Prompt: {response.usage.prompt_tokens}, "
                      f"Completion: {response.usage.completion_tokens}, "
                      f"Total: {response.usage.total_tokens}")

            # 获取响应文本
            text = response.choices[0].text.strip()

            if not text:
                print(f"⚠️ [WARNING] Empty response from API")
                if retry_count < self.max_retries - 1:
                    return self._call_api_completions(
                        prompt,
                        max_tokens=max_tokens * 2,
                        retry_count=retry_count + 1
                    )
                return self._fallback_text()

            print(f"✅ [SUCCESS] Got text response ({len(text)} chars)")
            return text

        except Exception as e:
            print(f"❌ [ERROR] API request failed: {e}")

            if retry_count < self.max_retries - 1:
                print(f"🔄 Retrying...")
                return self._call_api_completions(
                    prompt,
                    max_tokens=max_tokens,
                    retry_count=retry_count + 1
                )

            return self._fallback_text()

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        """
        Streaming API call for OpenAI
        """
        try:
            if not hasattr(self, 'client') or self.client is None:
                result = self._call_api(prompt, max_tokens, system_prompt)
                if result:
                    yield result
                return

            messages = [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            params = {
                "model": self.model,
                "messages": messages,
                "stream": True
            }

            if 'gpt-5' in self.model.lower():
                params['max_completion_tokens'] = max_tokens
                params['temperature'] = 1
            else:
                params['max_tokens'] = max_tokens
                params['temperature'] = 0.7

            response = self.client.chat.completions.create(**params)

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        except Exception as e:
            print(f"⚠️ OpenAI streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def _call_api_with_json_mode(
            self,
            prompt: str,
            max_tokens: int = 500,
            system_prompt: str = None,
            retry_count: int = 0,
            sampling: Optional[Dict] = None
    ) -> Dict:
        """
        使用 JSON 模式调用 OpenAI Chat Completions API

        🔧 如果是 codex 模型，自动路由到 completions 端点
        """

        # 🔑 关键修复：如果是 codex 模型，直接使用 completions 端点
        if self._is_codex_model(self.model):
            print(f"⚠️  [WARNING] _call_api_with_json_mode called with codex model, routing to completions endpoint")

            # 调用 completions API
            text_result = self._call_api_completions(prompt, max_tokens, retry_count)

            # 尝试解析为 JSON（如果已经是 JSON）
            try:
                return json.loads(text_result)
            except:
                # 不是 JSON，包装成 JSON 格式
                return {
                    "scenario": text_result,
                    "operation": "UNKNOWN",
                    "opcode": "0000",
                    "bitwidth": 16,
                    "note": "Generated via completions endpoint"
                }

        if retry_count >= self.max_retries:
            print(f"❌ [ERROR] Max retries ({self.max_retries}) reached")
            return self._fallback_json()

        try:
            max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
            if retry_count == 0:
                print(f"🔍 [DEBUG] JSON Mode Request - Model: {self.model}, max_tokens: {max_tokens}")
            else:
                print(f"🔄 [DEBUG] Retry {retry_count}/{self.max_retries} - max_tokens: {max_tokens}")

            # 构建消息（修复：原先只在 retry 分支构建，首次调用必抛 UnboundLocalError）
            default_system = """You are a helpful assistant that generates BDD scenario descriptions.
            You MUST respond with valid JSON only. Do not include any text outside the JSON structure."""

            system_content = system_prompt or default_system

            # OpenAI requires the word "json" in messages when using response_format json_object
            if 'json' not in system_content.lower() and 'json' not in prompt.lower():
                system_content += "\nYou MUST respond with valid JSON only."

            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # 根据模型选择参数
            if self._is_gpt5_model(self.model):
                # GPT-5 系列使用 max_completion_tokens
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                }
            else:
                # 旧模型使用 max_tokens
                completion_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
            completion_params.update(self._merge_sampling('json_mode', sampling)[0])
            completion_params["response_format"] = {"type": "json_object"}

            # 调用 API
            response = self.client.chat.completions.create(**completion_params)

            self._record_response_meta(response)

            # 调试信息
            choice = response.choices[0]
            content = choice.message.content

            print(f"🔗 [DEBUG] Response ID: {response.id}")
            print(f"🤖 [DEBUG] Model: {response.model}")
            print(f"📊 [DEBUG] finish_reason: {choice.finish_reason}, content_length: {len(content) if content else 0}")

            # 打印 token 使用情况
            if hasattr(response, 'usage') and response.usage:
                print(f"💰 [DEBUG] Tokens - "
                      f"Prompt: {response.usage.prompt_tokens}, "
                      f"Completion: {response.usage.completion_tokens}, "
                      f"Total: {response.usage.total_tokens}")

            # 检查响应
            if not content or not content.strip():
                print(f"⚠️ [WARNING] Empty response from API")
                return self._call_api_with_json_mode(
                    prompt,
                    max_tokens=max_tokens * 2,
                    system_prompt=system_prompt,
                    retry_count=retry_count + 1
                )

            # 解析 JSON
            try:
                result = json.loads(content)
                print(f"✅ [SUCCESS] Got valid JSON response")
                return result
            except json.JSONDecodeError as e:
                print(f"⚠️ [WARNING] JSON parse error: {e}")
                print(f"   Raw content: {content[:200]}...")

                # JSON 解析失败,重试
                if retry_count < self.max_retries - 1:
                    return self._call_api_with_json_mode(
                        prompt,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        retry_count=retry_count + 1
                    )
                else:
                    return self._fallback_json()

        except Exception as e:
            print(f"❌ [ERROR] API request failed: {e}")

            if retry_count < self.max_retries - 1:
                print(f"🔄 Retrying...")
                return self._call_api_with_json_mode(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    retry_count=retry_count + 1
                )

            return self._fallback_json()

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """生成 BDD 场景描述"""

        # 🔑 Codex 模型使用 completions 接口
        if self._is_codex_model(self.model):
            prompt = f"""Generate a BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style (Given-When-Then)
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:"""

            return self._call_api_completions(prompt, max_tokens=200)

        # 其他模型使用 chat completions 和 JSON mode
        prompt = f"""Generate a BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style (Given-When-Then)
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

IMPORTANT: You MUST respond with a JSON object in this exact format:
{{
  "scenario": "Your BDD scenario description here",
  "operation": "{operation_name}",
  "opcode": "{operation_code}",
  "bitwidth": {bitwidth}
}}

Respond with ONLY the JSON object, no other text.
"""

        result = self._call_api_with_json_mode(prompt, max_tokens=500)

        # 提取 scenario 字段
        if isinstance(result, dict) and "scenario" in result:
            return result["scenario"]
        else:
            print(f"⚠️ [WARNING] Unexpected JSON structure: {result}")
            if isinstance(result, dict):
                for key in ["scenario", "description", "text", "content"]:
                    if key in result:
                        return result[key]
            return "Given ALU operation, When executed, Then produce correct output"

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: list
    ) -> str:
        """生成 Feature 级别描述"""

        # 🔑 Codex 模型使用 completions 接口
        if self._is_codex_model(self.model):
            prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):"""

            return self._call_api_completions(prompt, max_tokens=200)

        # 其他模型使用 chat completions 和 JSON mode
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

IMPORTANT: You MUST respond with a JSON object in this exact format:
{{
  "feature": "Your Feature description here (2-4 sentences)",
  "bitwidth": {bitwidth},
  "operations_count": {operations_count}
}}

Respond with ONLY the JSON object, no other text.
"""

        result = self._call_api_with_json_mode(prompt, max_tokens=500)

        # 提取 feature 字段
        if isinstance(result, dict) and "feature" in result:
            return result["feature"]
        else:
            print(f"⚠️ [WARNING] Unexpected JSON structure: {result}")
            if isinstance(result, dict):
                for key in ["feature", "description", "text", "content"]:
                    if key in result:
                        return result[key]
            return f"Verification suite for {bitwidth}-bit ALU with {operations_count} operations"

    def _fallback_json(self, operation_name: str = "UNKNOWN", opcode: str = "0000", bitwidth: int = 16) -> Dict:
        """备用 JSON 响应，包含所有必需字段，使用默认值避免 KeyError"""
        return {
            "scenario": "Given ALU operation, When executed, Then produce correct output",
            "operation": operation_name,
            "opcode": opcode,
            "bitwidth": bitwidth,
            "error": "Fallback response due to API failure"
        }

    def _fallback_text(self) -> str:
        """备用文本响应 (用于 codex)"""
        return "Given ALU operation, When executed, Then produce correct output"

    def _call_api_text(self, prompt: str, max_tokens: int = 500, system_prompt: str = None,
                       sampling: Optional[Dict] = None) -> str:
        """
        纯文本 chat 调用（无 JSON mode）——benchmark 用。
        主应用的 intent 解析仍走 _call_api 的 JSON 模式，两者互不影响。
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
            if self._is_gpt5_model(self.model):
                params = {"model": self.model, "messages": messages,
                          "max_completion_tokens": max_tokens}
            else:
                params = {"model": self.model, "messages": messages,
                          "max_tokens": max_tokens}
            params.update(self._merge_sampling('text', sampling)[0])
            response = self.client.chat.completions.create(**params)
            self._record_response_meta(response)
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            print(f"⚠️  OpenAI text API request failed: {e}")
            return self._fallback_text()

    def _call_api(self, prompt: str, max_tokens: int = 500, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        """
        统一的 API 调用接口，返回纯文本——与其余 8 家 provider 一致。

        此前这里强制走 JSON mode 并返回 json.dumps(...)，于是生成的 Verilog
        经序列化后换行变成字面量 "\\n"，写进 .v 文件就是单行文本，iverilog
        直接 syntax error。基线批次里 OpenAI 的 DUV 编译通过率因此是 0/10。

        JSON mode 原本是为 intent 解析设计的，而那两个用途
        （generate_scenario_description / generate_feature_description）
        本来就直接调用 _call_api_with_json_mode，不经过这里，所以把本方法
        改回文本不影响它们。
        """
        print(f"   🔍 [DEBUG][OpenAI._call_api] Called with max_tokens={max_tokens}")

        # 🔑 Codex 模型使用 completions 接口（本就返回纯文本，不再包装成 JSON）
        if self._is_codex_model(self.model):
            return self._call_api_completions(prompt, max_tokens)

        return self._call_api_text(prompt, max_tokens, system_prompt, sampling=sampling)


class ClaudeProvider(LLMProvider):
    # Claude 当前不发送 temperature，沿用 API 自身默认值 -> DEFAULT_SAMPLING 留空
    # Anthropic Messages API 不支持 seed
    DEFAULT_SAMPLING = {}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'max_tokens'}
    # 无需覆盖 SAMPLING_RANGES：Anthropic Messages API 的 temperature 硬上界
    # 是 1.0，与基类统一采用的上界一致。

    """
    Anthropic Claude API Provider - PAID

    How to get API key:
    1. Visit: https://console.anthropic.com/
    2. Create account and add credits
    3. Get API key

    Models: claude-3-opus, claude-3-sonnet, claude-3-haiku
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            raise ValueError(
                "Claude API key not provided. Get key at: https://console.anthropic.com/")

    def _call_api(self, prompt: str, max_tokens: int = 200, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        """Call Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        # DEFAULT_SAMPLING 为空：不填时不发送 temperature，沿用 API 默认值
        payload.update(self._merge_sampling('default', sampling)[0])

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['content'][0]['text'].strip()
        except Exception as e:
            print(f"⚠️  Claude API request failed: {e}")
            return self._fallback_description(prompt)

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a BDD scenario description"""
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        """
        Streaming API call for Claude
        """
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        max_tokens, _ = self._resolve_max_tokens(max_tokens, sampling)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        payload.update(self._merge_sampling('stream', sampling)[0])

        if system_prompt:
            payload["system"] = system_prompt

        try:
            proxies = self._get_proxies()
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                proxies=proxies,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        try:
                            event = json.loads(data)
                            if event.get('type') == 'content_block_delta':
                                delta = event.get('delta', {})
                                text = delta.get('text', '')
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print(f"⚠️ Claude streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate a Feature-level description"""
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        """
        🔧 修复: Fallback 返回 JSON 格式以支持 intent 解析
        """
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class GrokProvider(LLMProvider):
    # xAI 为 OpenAI 兼容 API
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}

    """
    xAI Grok API Provider

    Large context window (131K tokens), suitable for complex CPU-level HDL generation.
    How to get API key:
    1. Visit: https://console.x.ai/
    2. Sign up and get API key

    Models: grok-3, grok-3-mini, grok-2
    API: OpenAI-compatible (v1/chat/completions)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-3-mini"):
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.model = model
        self.api_url = "https://api.x.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Grok API key not provided. Get key at: https://console.x.ai/")

    def _call_api(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('default', sampling)[0])

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120, proxies=proxies)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Grok API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, proxies=proxies, stream=True, timeout=300)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Grok streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_scenario_description(self, operation_name: str, operation_code: str, operation_description: str, bitwidth: int) -> str:
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(self, bitwidth: int, operations_count: int, operations_list: List[str]) -> str:
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class QwenProvider(LLMProvider):
    # DashScope compatible-mode 为 OpenAI 兼容 API
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}

    """
    Alibaba Qwen API Provider

    Strong code generation capabilities, free tier available.
    No proxy needed for Chinese users.
    How to get API key:
    1. Visit: https://dashscope.console.aliyun.com/
    2. Sign up and get API key

    Models: qwen-coder-plus, qwen-max, qwen-plus, qwen-turbo
    API: OpenAI-compatible (v1/chat/completions)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-coder-plus"):
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Qwen API key not provided. Get key at: https://dashscope.console.aliyun.com/")

    def _get_proxies(self) -> Dict[str, None]:
        """Qwen (Alibaba Cloud) 国内直连。
        注意: requests 里 proxies=None 表示"使用环境变量代理"，
        必须显式传 {'http': None, 'https': None} 才能真正绕过代理。"""
        return {'http': None, 'https': None}

    def _call_api(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('default', sampling)[0])

        try:
            response = requests.post(self.api_url, headers=headers, json=payload,
                                     proxies=self._get_proxies(), timeout=60)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  {type(self).__name__} API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            response = requests.post(self.api_url, headers=headers, json=payload,
                                     proxies=self._get_proxies(), stream=True, timeout=180)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️  {type(self).__name__} streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_scenario_description(self, operation_name: str, operation_code: str, operation_description: str, bitwidth: int) -> str:
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(self, bitwidth: int, operations_count: int, operations_list: List[str]) -> str:
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class MistralProvider(LLMProvider):
    # Mistral 的 seed 字段名是 random_seed
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}
    SAMPLING_FIELD_MAP = {'seed': 'random_seed'}

    """
    Mistral AI API Provider

    Strong code generation, Codestral model specialized for code tasks.
    How to get API key:
    1. Visit: https://console.mistral.ai/
    2. Sign up and get API key

    Models: codestral-latest, mistral-large-latest, mistral-small-latest
    API: OpenAI-compatible (v1/chat/completions)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "codestral-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Mistral API key not provided. Get key at: https://console.mistral.ai/")

    def _call_api(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('default', sampling)[0])

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60, proxies=proxies)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Mistral API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, proxies=proxies, stream=True, timeout=180)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Mistral streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_scenario_description(self, operation_name: str, operation_code: str, operation_description: str, bitwidth: int) -> str:
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(self, bitwidth: int, operations_count: int, operations_list: List[str]) -> str:
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class TogetherProvider(LLMProvider):
    # Together 为 OpenAI 兼容 API
    DEFAULT_SAMPLING = {'default': {'temperature': 0.7},
                        'stream': {'temperature': 0.7}}
    SUPPORTED_SAMPLING = {'temperature', 'top_p', 'seed', 'max_tokens'}

    """
    Together AI API Provider

    Aggregation platform with access to many open-source models via one API.
    How to get API key:
    1. Visit: https://api.together.xyz/
    2. Sign up and get API key

    Models: meta-llama/Llama-3.3-70B-Instruct-Turbo, deepseek-ai/DeepSeek-V3,
            Qwen/Qwen2.5-Coder-32B-Instruct, mistralai/Mixtral-8x22B-Instruct-v0.1
    API: OpenAI-compatible (v1/chat/completions)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.model = model
        self.api_url = "https://api.together.xyz/v1/chat/completions"

        if not self.api_key:
            raise ValueError("Together AI API key not provided. Get key at: https://api.together.xyz/")

    def _call_api(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                  sampling: Optional[Dict] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a helpful assistant that generates clear, concise BDD scenario descriptions for hardware verification."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('default', sampling)[0])

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60, proxies=proxies)
            response.raise_for_status()
            result = response.json()
            self._record_response_meta(result)
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"⚠️  Together AI API request failed: {e}")
            return self._fallback_description(prompt)

    def _call_api_stream(self, prompt: str, max_tokens: int = 4000, system_prompt: str = None,
                         sampling: Optional[Dict] = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or "You are a hardware verification expert."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self._resolve_max_tokens(max_tokens, sampling)[0],
        }
        payload.update(self._merge_sampling('stream', sampling)[0])
        payload["stream"] = True

        try:
            proxies = self._get_proxies()
            response = requests.post(self.api_url, headers=headers, json=payload, proxies=proxies, stream=True, timeout=180)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                # 流式的 finish_reason 只在最后一个 chunk 非空，
                                # 截断判定（finish_reason == 'length'）依赖它
                                self._record_response_meta(chunk_data)
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Together AI streaming failed: {e}")
            result = self._call_api(prompt, max_tokens, system_prompt)
            if result:
                yield result

    def generate_scenario_description(self, operation_name: str, operation_code: str, operation_description: str, bitwidth: int) -> str:
        prompt = f"""Generate a clear BDD scenario description for testing a {bitwidth}-bit ALU operation.

Operation Details:
- Name: {operation_name}
- Opcode: {operation_code}
- Description: {operation_description}
- Bitwidth: {bitwidth} bits

Requirements:
1. Write in Gherkin/BDD style
2. Be specific about the operation being tested
3. Keep it concise (1-2 sentences)
4. Focus on functional behavior

Generate the scenario description:
"""
        return self._call_api(prompt, max_tokens=150)

    def generate_feature_description(self, bitwidth: int, operations_count: int, operations_list: List[str]) -> str:
        prompt = f"""Generate a BDD Feature description for a {bitwidth}-bit ALU verification suite.

ALU Details:
- Bitwidth: {bitwidth} bits
- Total Operations: {operations_count}
- Operations: {', '.join(operations_list[:10])}

Generate a concise Feature description (2-4 sentences):
"""
        return self._call_api(prompt, max_tokens=200)

    def _fallback_description(self, prompt: str) -> str:
        if "JSON" in prompt or "json" in prompt or '"operation"' in prompt:
            print("   🔧 [FALLBACK] Returning JSON format for intent parsing")
            return self._fallback_intent_json(prompt)
        return "Test ALU operation with various input values and verify correct output"


class LocalLLMProvider(LLMProvider):
    """
    Local Template Provider - FREE and NO API NEEDED!

    Uses predefined templates for generating descriptions.
    No internet connection or API key required.
    """

    def __init__(self):
        pass

    def generate_scenario_description(
            self,
            operation_name: str,
            operation_code: str,
            operation_description: str,
            bitwidth: int
    ) -> str:
        """Generate a simple template-based scenario description"""
        return f"""As a verification engineer
I want to test the {operation_name} operation (opcode: {operation_code})
So that I can verify the {bitwidth}-bit ALU correctly performs {operation_description.lower()}"""

    def generate_feature_description(
            self,
            bitwidth: int,
            operations_count: int,
            operations_list: List[str]
    ) -> str:
        """Generate Feature-level description"""
        return f"""As a hardware verification engineer
I want to verify the {bitwidth}-bit ALU implementation
So that I can ensure it correctly performs all {operations_count} supported arithmetic and logical operations,
including {', '.join(operations_list[:5])}{' and more' if len(operations_list) > 5 else ''}"""


class LlamaProvider(QwenProvider):
    """Meta Llama，走与 Qwen 相同的 OpenAI 兼容端点、同一把 API key。

    继承 QwenProvider 是因为两者的请求/响应格式、采样参数映射完全一致，
    只有默认模型不同。单独立为一个 provider（而不是 Qwen 的一个 model 选项）
    是为了在实验数据里独立成一家：llm_calls.provider 会记成 LlamaProvider，
    跨模型对比时不会和 Qwen 混在一起。

    Key：优先 LLAMA_API_KEY，缺失时回退 QWEN_API_KEY / DASHSCOPE_API_KEY，
    所以只配一处也能用。
    端点：默认与 Qwen 相同，可用 LLAMA_API_URL 覆盖（换服务商时不必改代码）。
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "llama-3.3-70b-instruct"):
        api_key = (api_key or os.getenv("LLAMA_API_KEY")
                   or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
        if not api_key:
            raise ValueError(
                "Llama API key not provided. It shares the key with Qwen — set "
                "LLAMA_API_KEY, QWEN_API_KEY or DASHSCOPE_API_KEY.")
        super().__init__(api_key=api_key, model=model)
        # 默认走国际站：探测确认 llama-3.3-70b-instruct 在国内站返回 404
        # (model does not exist)，在国际站返回 401 (incorrect API key) —— 也就是
        # 国际站认得这个模型，只是需要有效 key。换服务商时用 LLAMA_API_URL 覆盖。
        self.api_url = os.getenv(
            "LLAMA_API_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")


# ========== FACTORY ==========


class LLMFactory:
    """Factory class for creating LLM providers"""

    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """Create an LLM provider instance"""
        providers = {
            # FREE providers
            'gemini': GeminiProvider,
            'google': GeminiProvider,
            'groq': GroqProvider,
            'deepseek': DeepSeekProvider,
            'qwen': QwenProvider,
            'tongyi': QwenProvider,
            'llama': LlamaProvider,
            'meta': LlamaProvider,
            'local': LocalLLMProvider,

            # PAID providers
            'openai': OpenAIProvider,
            'gpt': OpenAIProvider,
            'chatgpt': OpenAIProvider,
            'claude': ClaudeProvider,
            'anthropic': ClaudeProvider,
            'grok': GrokProvider,
            'xai': GrokProvider,
            'mistral': MistralProvider,
            'codestral': MistralProvider,
            'together': TogetherProvider,
            'together_ai': TogetherProvider,
        }

        provider_type = provider_type.lower()
        if provider_type not in providers:
            print(f"⚠️  Unknown provider type: {provider_type}, using local template provider")
            provider_type = 'local'

        # 未显式指定 model 时，采用配置文件里为该 provider 设定的模型
        # （由 main.load_config 导出成 {NAME}_MODEL）。没有配置就退回类默认值。
        if 'model' not in kwargs or not kwargs.get('model'):
            canonical = {
                'google': 'gemini', 'tongyi': 'qwen', 'meta': 'llama',
                'gpt': 'openai', 'chatgpt': 'openai', 'anthropic': 'claude',
                'xai': 'grok', 'codestral': 'mistral', 'together_ai': 'together',
            }.get(provider_type, provider_type)
            configured = os.getenv(f"{canonical.upper()}_MODEL")
            if configured:
                kwargs['model'] = configured

        try:
            provider = providers[provider_type](**kwargs)

            # Print provider info
            if provider_type in ['gemini', 'google']:
                if GENAI_AVAILABLE:
                    print("🆓 Using Google Gemini (FREE) with SDK - retry & fallback enabled")
                else:
                    print("🆓 Using Google Gemini (FREE) with REST API")
            elif provider_type == 'groq':
                print("🆓 Using Groq (FREE and FAST)")
            elif provider_type == 'deepseek':
                print("🆓 Using DeepSeek (FREE)")
            elif provider_type == 'local':
                print("📝 Using Local Templates (FREE)")
            elif provider_type in ['openai', 'gpt', 'chatgpt']:
                model = kwargs.get('model', 'gpt-5-mini')
                if 'gpt-5' in model:
                    if 'codex' in model.lower():
                        print(f"💰 Using OpenAI GPT-5 Codex (PAID) - Model: {model} [Completions API]")
                    else:
                        print(f"💰 Using OpenAI GPT-5 Series (PAID) - Model: {model}")
                else:
                    print(f"💰 Using OpenAI (PAID) - Model: {model}")
            elif provider_type in ['claude', 'anthropic']:
                print("💰 Using Claude (PAID)")
            elif provider_type in ['grok', 'xai']:
                print("💰 Using Grok/xAI (131K context, ideal for complex HDL)")
            elif provider_type in ['qwen', 'tongyi']:
                print("🆓 Using Qwen (FREE tier, no proxy needed)")
            elif provider_type in ['mistral', 'codestral']:
                print("💰 Using Mistral/Codestral (code-specialized)")
            elif provider_type in ['together', 'together_ai']:
                print("🆓 Using Together AI (multi-model platform)")

            return provider
        except Exception as e:
            print(f"⚠️  Failed to create provider {provider_type}: {e}")
            print("🔄 Falling back to local provider")
            return LocalLLMProvider()

    @staticmethod
    def list_providers() -> Dict[str, Dict[str, str]]:
        """List all available providers with their details"""
        return {
            "FREE": {
                "gemini": "Google Gemini - 60 req/min (Get key: https://makersuite.google.com/app/apikey)",
                "groq": "Groq - Fast and free (Get key: https://console.groq.com/keys)",
                "deepseek": "DeepSeek - Chinese LLM (Get key: https://platform.deepseek.com/)",
                "qwen": "Qwen - Alibaba, strong code gen (Get key: https://dashscope.console.aliyun.com/)",
                "together": "Together AI - Multi-model platform (Get key: https://api.together.xyz/)",
                "local": "Local Templates - No API needed"
            },
            "PAID": {
                "openai": "OpenAI GPT (GPT-5 series including gpt-5.1-codex) - Requires credits",
                "claude": "Anthropic Claude - Requires credits",
                "grok": "xAI Grok - 131K context, ideal for complex HDL (Get key: https://console.x.ai/)",
                "mistral": "Mistral/Codestral - Code-specialized (Get key: https://console.mistral.ai/)"
            }
        }


class LLMConfig:
    """Configuration manager for LLM providers"""

    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}")

        return {
            "provider": "local",
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key": ""
            },
            "groq": {
                "model": "mixtral-8x7b-32768",
                "api_key": ""
            },
            "deepseek": {
                "model": "deepseek-chat",
                "api_key": ""
            },
            "openai": {
                "model": "gpt-5-mini",
                "api_key": ""
            },
            "claude": {
                "model": "claude-sonnet-4-20250514",
                "api_key": ""
            },
            "grok": {
                "model": "grok-3-mini",
                "api_key": ""
            },
            "qwen": {
                "model": "qwen-coder-plus",
                "api_key": ""
            },
            "mistral": {
                "model": "codestral-latest",
                "api_key": ""
            },
            "together": {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "api_key": ""
            }
        }

    def save_config(self):
        """Save configuration file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")

    def get_provider(self) -> LLMProvider:
        """Get configured LLM provider"""
        provider_type = self.config.get("provider", "local")

        kwargs = {}
        if provider_type in self.config:
            provider_config = self.config[provider_type]
            if "api_key" in provider_config:
                kwargs["api_key"] = provider_config["api_key"]
            if "model" in provider_config:
                kwargs["model"] = provider_config["model"]

        return LLMFactory.create_provider(provider_type, **kwargs)


if __name__ == '__main__':
    print("🧪 Testing LLM Provider Module (GPT-5 series including gpt-5.1-codex)\n")
    print("=" * 70)

    print("\n📋 Available Providers:")
    providers = LLMFactory.list_providers()

    print("\n🆓 FREE Providers:")
    for name, desc in providers["FREE"].items():
        print(f"   • {name}: {desc}")

    print("\n💰 PAID Providers:")
    for name, desc in providers["PAID"].items():
        print(f"   • {name}: {desc}")

    print("\n" + "=" * 70)
    print("\n1️⃣  Testing Local Provider:")
    local_provider = LocalLLMProvider()
    desc = local_provider.generate_scenario_description(
        "ADD", "0000", "Addition (A + B)", 16
    )
    print(f"   Scenario description: {desc}\n")

    feature_desc = local_provider.generate_feature_description(
        16, 12, ["ADD", "SUB", "AND", "OR", "XOR"]
    )
    print(f"   Feature description:\n{feature_desc}\n")

    print("2️⃣  Testing Configuration Manager:")
    config = LLMConfig()
    print(f"   Current provider: {config.config.get('provider')}")
    print(f"   Default OpenAI model: {config.config.get('openai', {}).get('model')}")
    provider = config.get_provider()
    print(f"   Provider type: {type(provider).__name__}\n")

    # 🔧 测试 fallback JSON 生成
    print("3️⃣  Testing Fallback JSON Generation:")
    test_prompt = '''Analyze this test scenario request and extract information in JSON format.
User Request: "Create ADD scenario with A = B, 3 examples"
Extract:
1. "operation": ALU operation (ADD, SUB, AND, OR, XOR, NOT, SHL, SHR)
...'''

    fallback_json = local_provider._fallback_intent_json(test_prompt)
    print(f"   Fallback JSON:\n{fallback_json}\n")

    # 如果有 OpenAI API key,测试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("4️⃣  Testing OpenAI Provider with gpt-5.1-codex:")
        print("-" * 70)
        try:
            openai_provider = LLMFactory.create_provider("openai", model="gpt-5.1-codex")
            desc = openai_provider.generate_scenario_description(
                "XOR", "0100", "XOR operation (A XOR B)", 16
            )
            print(f"\n✅ Scenario Result:")
            print(f"   {desc}\n")
        except Exception as e:
            print(f"   Failed: {e}\n")
    else:
        print("4️⃣  OpenAI API Key not found. Set OPENAI_API_KEY to test.")

    print("=" * 70)
    print("✅ Testing completed!")
    print("\n💡 To use gpt-5.1-codex:")
    print("   python script.py --llm-provider openai --model gpt-5.1-codex")
    print("\n🔧 Key Implementation:")
    print("   ✓ gpt-5.1-codex uses completions endpoint (v1/completions)")
    print("   ✓ Other GPT-5 models use chat completions (v1/chat/completions)")
    print("   ✓ Codex returns plain text; others use JSON mode")
    print("   ✓ Fallback now returns JSON format for intent parsing")