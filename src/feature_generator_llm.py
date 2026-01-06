"""
Feature Generator (LLM-based) - Generate BDD Feature files using LLMs
=====================================================================

This module uses LLMs to directly generate complete BDD Feature files,
including test cases. This replaces the previous random test generation.

NEW ARCHITECTURE:
- LLM generates Feature file directly (not just specs)
- Test cases come from LLM (not random generator)
- Supports flexible user input (simple to detailed)

DIRECTORY STRUCTURE:
    output/
    └── bdd/
        ├── groq/
        │   └── alu_16bit_xxx.feature  ← LLM generates this
        ├── deepseek/
        └── openai/

USAGE:
    # Simple
    python feature_generator_llm.py -i "16-bit ALU"

    # Medium
    python feature_generator_llm.py -i "32-bit ALU with ADD, SUB, XOR"

    # Detailed
    python feature_generator_llm.py -i "16-bit ALU: ADD(0000), SUB(0001), 8 tests each"
"""

from __future__ import annotations  # Enable postponed evaluation of annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# ============================================================================
# Proxy Setup (same as spec_generator)
# ============================================================================
def setup_proxy():
    """Setup proxy from config file for international APIs"""
    config_paths = [
        Path('config/llm_config.json'),
        Path('llm_config.json'),
        Path('../config/llm_config.json'),
        Path(__file__).parent / 'config' / 'llm_config.json',
        Path(__file__).parent.parent / 'config' / 'llm_config.json',
    ]

    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        proxy_config = config.get('proxy', {})
        if proxy_config.get('enabled'):
            http_proxy = proxy_config.get('http_proxy', '')
            https_proxy = proxy_config.get('https_proxy', '')
            os.environ['HTTP_PROXY'] = http_proxy
            os.environ['HTTPS_PROXY'] = https_proxy
            print(f"🌐 Proxy enabled: {https_proxy}")
    except Exception as e:
        print(f"⚠️  Proxy config error: {e}")

setup_proxy()
# ============================================================================

# Import LLM providers
try:
    from llm_providers import (
        LLMProvider,
        LLMFactory,
        GroqProvider,
        DeepSeekProvider,
        OpenAIProvider,
        ClaudeProvider,
        GeminiProvider,
        LocalLLMProvider
    )
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️  Warning: llm_providers module not found")
    print("   Feature generation will be limited")


class FeatureGeneratorLLM:
    """
    Generate BDD Feature files using LLMs

    This replaces the old spec → random tests → feature pipeline
    with a direct LLM → feature generation approach.
    """

    # Map LLM class names to standard names
    LLM_NAMES = {
        'GroqProvider': 'groq',
        'DeepSeekProvider': 'deepseek',
        'OpenAIProvider': 'openai',
        'ClaudeProvider': 'claude',
        'GeminiProvider': 'gemini',
        'LocalLLMProvider': 'local',
    }

    # Default operations for different levels
    DEFAULT_OPERATIONS = {
        'basic': ['ADD', 'SUB', 'AND', 'OR'],
        'extended': ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'NOT'],
        'full': ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'NOT', 'SHL', 'SHR']
    }

    # Default opcodes
    DEFAULT_OPCODES = {
        'ADD': '0000', 'SUB': '0001', 'AND': '0010', 'OR': '0011',
        'XOR': '0100', 'NOT': '0101', 'SHL': '0110', 'SHR': '0111'
    }

    def __init__(
        self,
        llm_provider: str = 'groq',
        output_dir: Optional[str] = None,
        project_root: Optional[str] = None,
        debug: bool = True
    ):
        """
        Initialize Feature Generator

        Args:
            llm_provider: LLM provider name (groq, deepseek, openai, etc.)
            output_dir: Output base directory
            project_root: Project root directory
            debug: Enable debug output
        """
        self.debug = debug
        self.llm_provider_name = llm_provider

        # Initialize LLM
        if HAS_LLM:
            self.llm = LLMFactory.create_provider(llm_provider)
            self.llm_name = self._detect_llm_name()
        else:
            self.llm = None
            self.llm_name = 'local'

        # Setup output paths
        self._setup_paths(project_root, output_dir)

        print(f"🚀 Feature Generator (LLM-based)")
        print(f"   LLM Provider: {self.llm_name}")
        print(f"   Output: {self.output_dir}")

    def _detect_llm_name(self) -> str:
        """Detect LLM provider name from instance"""
        if self.llm is None:
            return 'local'
        class_name = type(self.llm).__name__
        return self.LLM_NAMES.get(class_name, 'unknown')

    def _setup_paths(self, project_root: Optional[str], output_dir: Optional[str]):
        """Setup output directory paths"""
        if output_dir:
            base_dir = Path(output_dir)
        elif project_root:
            base_dir = Path(project_root) / "output" / "bdd"
        else:
            current = Path.cwd()
            possible_paths = [
                current / "output" / "bdd",
                current / "bdd",
                current.parent / "output" / "bdd",
            ]

            for path in possible_paths:
                if path.exists():
                    base_dir = path
                    break
            else:
                base_dir = current / "output" / "bdd"

        # Create LLM-specific subdirectory
        self.output_dir = base_dir / self.llm_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_user_input(self, user_input: str) -> Dict:
        """
        Parse user input to extract requirements

        Supports three levels:
        - Simple: "16-bit ALU"
        - Medium: "32-bit ALU with ADD, SUB, XOR"
        - Detailed: "16-bit ALU: ADD(0000), SUB(0001), 8 tests per op"

        Args:
            user_input: Natural language input

        Returns:
            Dict with parsed requirements
        """
        input_lower = user_input.lower()

        # Extract bitwidth
        bitwidth = 16  # default
        bitwidth_patterns = [
            (r'(\d+)[\s-]*bit', lambda m: int(m.group(1))),
            (r'(\d+)b\s', lambda m: int(m.group(1))),
        ]
        for pattern, extract in bitwidth_patterns:
            match = re.search(pattern, input_lower)
            if match:
                bitwidth = extract(match)
                break

        # Validate bitwidth
        if bitwidth not in [8, 16, 32, 64]:
            print(f"⚠️  Invalid bitwidth: {bitwidth}, using 16")
            bitwidth = 16

        # Extract module type
        module_type = 'alu'  # default
        if 'counter' in input_lower:
            module_type = 'counter'
        elif 'regfile' in input_lower or 'register' in input_lower:
            module_type = 'regfile'
        elif 'cpu' in input_lower or 'risc' in input_lower or 'riscv' in input_lower:
            module_type = 'cpu'

        # Extract depth for regfile (e.g., "8x", "16 registers", "8x8")
        depth = 16  # default
        if module_type == 'regfile':
            depth_patterns = [
                (r'(\d+)x\d+', lambda m: int(m.group(1))),  # "8x8", "32x32"
                (r'(\d+)\s*registers?', lambda m: int(m.group(1))),  # "8 registers"
                (r'(\d+)x[\s\-]', lambda m: int(m.group(1))),  # "16x " or "16x-"
            ]
            for pattern, extract in depth_patterns:
                match = re.search(pattern, input_lower)
                if match:
                    depth = extract(match)
                    break

        # Extract pipeline stages for CPU (e.g., "5-stage", "3 stage")
        pipeline_stages = 5  # default
        if module_type == 'cpu':
            pipeline_match = re.search(r'(\d+)[\s-]*stage', input_lower)
            if pipeline_match:
                pipeline_stages = int(pipeline_match.group(1))
                if pipeline_stages not in [3, 5]:
                    print(f"⚠️  Invalid pipeline stages: {pipeline_stages}, using 5")
                    pipeline_stages = 5

        # Extract operations
        operations = []
        operation_names = ['ADD', 'SUB', 'AND', 'OR', 'XOR', 'NOT', 'SHL', 'SHR', 'NAND', 'NOR']
        for op in operation_names:
            if op.lower() in input_lower:
                operations.append(op)

        # If no operations specified, use default
        if not operations:
            operations = self.DEFAULT_OPERATIONS['basic']

        # Extract opcodes (if specified)
        opcodes = {}
        opcode_pattern = r'(\w+)\s*\((\d{4})\)'
        for match in re.finditer(opcode_pattern, user_input):
            op_name = match.group(1).upper()
            opcode = match.group(2)
            if op_name in operations:
                opcodes[op_name] = opcode

        # Fill in missing opcodes with defaults
        for op in operations:
            if op not in opcodes:
                opcodes[op] = self.DEFAULT_OPCODES.get(op, '0000')

        # Extract number of tests
        num_tests = 5  # default
        num_tests_pattern = r'(\d+)\s*tests?'
        match = re.search(num_tests_pattern, input_lower)
        if match:
            num_tests = int(match.group(1))
            num_tests = max(5, min(num_tests, 20))  # Limit: 5-20

        return {
            'bitwidth': bitwidth,
            'module_type': module_type,
            'operations': operations,
            'opcodes': opcodes,
            'num_tests': num_tests,
            'original_input': user_input,
            'pipeline_stages': pipeline_stages,
            'depth': depth
        }

    def generate_feature(self, user_input: str) -> Optional[str]:
        """
        Generate Feature file from user input

        Args:
            user_input: User requirements

        Returns:
            Path to generated feature file
        """
        print(f"\n{'='*70}")
        print(f"🔍 Processing: {user_input}")
        print(f"{'='*70}\n")

        # Parse input
        requirements = self.parse_user_input(user_input)

        print(f"✅ Parsed requirements:")
        print(f"   Bitwidth: {requirements['bitwidth']}-bit")
        print(f"   Operations: {', '.join(requirements['operations'])}")
        print(f"   Tests per operation: {requirements['num_tests']}")
        print()

        # Generate prompt
        prompt = self._create_prompt(requirements)

        # Call LLM
        print(f"📡 Calling {self.llm_name} LLM...")
        feature_content = self._call_llm(prompt)

        if not feature_content:
            print("❌ Failed to generate feature")
            return None

        # Save feature file
        feature_path = self._save_feature(feature_content, requirements)

        print(f"\n✅ Feature file generated:")
        print(f"   {feature_path}")

        return str(feature_path)

    def _create_prompt(self, req: Dict) -> str:
        """
        Create prompt for LLM based on module type

        Supports: ALU, Counter, RegFile, CPU
        """
        module_type = req.get('module_type', 'alu')

        if module_type == 'counter':
            return self._create_counter_prompt(req)
        elif module_type == 'regfile':
            return self._create_regfile_prompt(req)
        elif module_type == 'cpu':
            return self._create_cpu_prompt(req)
        else:
            return self._create_alu_prompt(req)

    def _create_alu_prompt(self, req: Dict) -> str:
        """Create prompt for ALU Feature file"""
        max_value = (1 << req['bitwidth']) - 1

        # Build operations string with opcodes
        ops_list = []
        for op in req['operations']:
            opcode = req['opcodes'].get(op, '0000')
            ops_list.append(f"{op} (opcode: {opcode})")
        ops_str = '\n- '.join(ops_list)

        prompt = f"""You are an expert in hardware verification and BDD (Behavior-Driven Development).
    Generate a Gherkin-format Feature file for ALU testing.

    CRITICAL FORMAT REQUIREMENTS:
    1. Use strict Gherkin syntax with Scenario Outline and Examples tables
    2. Examples table MUST use pipe (|) separators with consistent spacing
    3. Each operation MUST have at least {req['num_tests']} test cases
    4. MUST include these specific edge cases:
       - Test with (0, 0)
       - Test with ({max_value}, {max_value}) - maximum values
       - Test cases that trigger overflow
    5. Follow the EXACT format shown below

    EXAMPLE FORMAT (8-bit):

    Feature: 8-bit ALU Verification
      As a hardware verification engineer
      I want to verify the ALU implementation
      So that I can ensure correct operation

      Background:
        Given the ALU is initialized with 8-bit operands

      @add @arithmetic
      Scenario Outline: Verify ADD operation
        Given I have operand A = <A>
        And I have operand B = <B>
        When I perform the ADD operation with opcode 0000
        Then the result should be <Expected_Result>
        And the zero flag should be <Zero_Flag>
        And the overflow flag should be <Overflow>
        And the negative flag should be <Negative_Flag>

        Examples:
          | A   | B   | Opcode | Expected_Result | Zero_Flag | Overflow | Negative_Flag |
          | 0   | 0   | 0000   | 0               | True      | False    | False         |
          | 0   | 1   | 0000   | 1               | False     | False    | False         |
          | 255 | 0   | 0000   | 255             | False     | False    | True          |
          | 255 | 255 | 0000   | 254             | False     | True     | True          |
          | 100 | 50  | 0000   | 150             | False     | False    | True          |

    NOW GENERATE:
    - Bitwidth: {req['bitwidth']}-bit
    - Operations to include:
    - {ops_str}
    - Each operation needs {req['num_tests']} test cases minimum
    - Maximum value for {req['bitwidth']}-bit: {max_value}
    - MUST include edge cases: (0,0), ({max_value},{max_value}), and overflow tests

    Generate the complete Feature file following the exact format above.
    Output ONLY the Feature file content, no explanations.
    """
        return prompt

    def _create_counter_prompt(self, req: Dict) -> str:
        """Create prompt for Counter Feature file"""
        max_value = (1 << req['bitwidth']) - 1

        prompt = f"""You are an expert in hardware verification and BDD (Behavior-Driven Development).
    Generate a Gherkin-format Feature file for Counter testing.

    CRITICAL FORMAT REQUIREMENTS:
    1. Use strict Gherkin syntax with Scenario Outline and Examples tables
    2. Examples table MUST use pipe (|) separators with consistent spacing
    3. Test all counter modes: UP (00), DOWN (01), UP-DOWN (10)
    4. Include edge cases: overflow, underflow, zero crossing

    EXAMPLE FORMAT:

    Feature: {req['bitwidth']}-bit Counter Verification
      As a hardware verification engineer
      I want to verify the Counter implementation
      So that I can ensure correct counting behavior

      Background:
        Given the Counter is initialized with {req['bitwidth']}-bit width

      @up_count
      Scenario Outline: Verify UP counting mode
        Given the counter is in UP mode (mode=00)
        And the counter value is <Initial_Value>
        When I enable counting for <Cycles> cycles
        Then the counter value should be <Expected_Value>
        And the overflow flag should be <Overflow>

        Examples:
          | Initial_Value | Cycles | Expected_Value | Overflow |
          | 0             | 1      | 1              | False    |
          | 0             | 5      | 5              | False    |
          | {max_value - 1} | 1      | {max_value}    | False    |
          | {max_value}   | 1      | 0              | True     |

      @down_count
      Scenario Outline: Verify DOWN counting mode
        Given the counter is in DOWN mode (mode=01)
        And the counter value is <Initial_Value>
        When I enable counting for <Cycles> cycles
        Then the counter value should be <Expected_Value>
        And the underflow flag should be <Underflow>

        Examples:
          | Initial_Value | Cycles | Expected_Value | Underflow |
          | 5             | 1      | 4              | False     |
          | 1             | 1      | 0              | False     |
          | 0             | 1      | {max_value}    | True      |

    NOW GENERATE:
    - Bitwidth: {req['bitwidth']}-bit
    - Maximum value: {max_value}
    - Modes: UP (00), DOWN (01), UP-DOWN (10)
    - Include at least {req['num_tests']} test cases per mode

    Generate the complete Feature file following the exact format above.
    Output ONLY the Feature file content, no explanations.
    """
        return prompt

    def _create_regfile_prompt(self, req: Dict) -> str:
        """Create prompt for Register File Feature file"""
        max_value = (1 << req['bitwidth']) - 1
        depth = req.get('depth', 16)
        addr_width = (depth - 1).bit_length()

        prompt = f"""You are an expert in hardware verification and BDD (Behavior-Driven Development).
    Generate a Gherkin-format Feature file for Register File testing.

    CRITICAL FORMAT REQUIREMENTS:
    1. Use strict Gherkin syntax with Scenario Outline and Examples tables
    2. Examples table MUST use pipe (|) separators with consistent spacing
    3. Test read/write operations
    4. Test that register 0 always reads as 0 (RISC-V style)

    EXAMPLE FORMAT:

    Feature: {depth}x{req['bitwidth']}-bit Register File Verification
      As a hardware verification engineer
      I want to verify the Register File implementation
      So that I can ensure correct read/write behavior

      Background:
        Given the Register File has {depth} registers of {req['bitwidth']}-bit width

      @write_read
      Scenario Outline: Verify write and read back
        Given I write <Write_Data> to register <Reg_Addr>
        When I read from register <Reg_Addr>
        Then the read data should be <Expected_Data>

        Examples:
          | Reg_Addr | Write_Data | Expected_Data |
          | 1        | 100        | 100           |
          | 2        | 255        | 255           |
          | 5        | 0          | 0             |
          | 10       | {max_value}| {max_value}   |

      @reg_zero
      Scenario Outline: Verify register 0 always returns 0
        Given I write <Write_Data> to register 0
        When I read from register 0
        Then the read data should be 0

        Examples:
          | Write_Data |
          | 100        |
          | {max_value}|
          | 1          |

      @dual_read
      Scenario Outline: Verify dual port read
        Given I write <Data1> to register <Reg1>
        And I write <Data2> to register <Reg2>
        When I read from register <Reg1> on port 1
        And I read from register <Reg2> on port 2
        Then port 1 should return <Data1>
        And port 2 should return <Data2>

        Examples:
          | Reg1 | Data1 | Reg2 | Data2 |
          | 1    | 100   | 2    | 200   |
          | 5    | 50    | 10   | 100   |

    NOW GENERATE:
    - Bitwidth: {req['bitwidth']}-bit
    - Depth: {depth} registers
    - Address width: {addr_width} bits
    - Maximum data value: {max_value}
    - Include at least {req['num_tests']} test cases per scenario type

    Generate the complete Feature file following the exact format above.
    Output ONLY the Feature file content, no explanations.
    """
        return prompt

    def _create_cpu_prompt(self, req: Dict) -> str:
        """Create prompt for RISC-V CPU Feature file"""
        pipeline_stages = req.get('pipeline_stages', 3)

        prompt = f"""You are an expert in hardware verification and BDD (Behavior-Driven Development).
    Generate a Gherkin-format Feature file for RISC-V CPU testing.

    CRITICAL FORMAT REQUIREMENTS:
    1. Use strict Gherkin syntax with Scenario Outline and Examples tables
    2. Examples table MUST use pipe (|) separators with consistent spacing
    3. Test basic RV32I instructions
    4. Test pipeline behavior and hazards

    EXAMPLE FORMAT:

    Feature: RISC-V CPU Verification ({pipeline_stages}-stage pipeline)
      As a hardware verification engineer
      I want to verify the RISC-V CPU implementation
      So that I can ensure correct instruction execution

      Background:
        Given the CPU is a {pipeline_stages}-stage pipeline RV32I processor
        And all registers are initialized to 0

      @arithmetic
      Scenario Outline: Verify arithmetic instructions
        Given register x<Rs1> contains <Rs1_Value>
        And register x<Rs2> contains <Rs2_Value>
        When I execute <Instruction>
        Then register x<Rd> should contain <Expected>

        Examples:
          | Instruction          | Rs1 | Rs1_Value | Rs2 | Rs2_Value | Rd | Expected |
          | add x1, x2, x3       | 2   | 10        | 3   | 20        | 1  | 30       |
          | sub x1, x2, x3       | 2   | 30        | 3   | 10        | 1  | 20       |
          | addi x1, x2, 100     | 2   | 50        | 0   | 0         | 1  | 150      |

      @logical
      Scenario Outline: Verify logical instructions
        Given register x<Rs1> contains <Rs1_Value>
        And register x<Rs2> contains <Rs2_Value>
        When I execute <Instruction>
        Then register x<Rd> should contain <Expected>

        Examples:
          | Instruction          | Rs1 | Rs1_Value | Rs2 | Rs2_Value | Rd | Expected |
          | and x1, x2, x3       | 2   | 0xFF      | 3   | 0x0F      | 1  | 0x0F     |
          | or x1, x2, x3        | 2   | 0xF0      | 3   | 0x0F      | 1  | 0xFF     |
          | xor x1, x2, x3       | 2   | 0xFF      | 3   | 0xFF      | 1  | 0x00     |

      @memory
      Scenario Outline: Verify load/store instructions
        Given register x<Rs1> contains <Address>
        And register x<Rs2> contains <Store_Data>
        When I execute store: sw x<Rs2>, 0(x<Rs1>)
        And I execute load: lw x<Rd>, 0(x<Rs1>)
        Then register x<Rd> should contain <Store_Data>

        Examples:
          | Rs1 | Address | Rs2 | Store_Data | Rd |
          | 2   | 0x100   | 3   | 0x12345678 | 1  |
          | 2   | 0x200   | 3   | 0xDEADBEEF | 1  |

      @branch
      Scenario Outline: Verify branch instructions
        Given register x<Rs1> contains <Rs1_Value>
        And register x<Rs2> contains <Rs2_Value>
        And PC is at <Initial_PC>
        When I execute <Instruction>
        Then PC should be <Expected_PC>

        Examples:
          | Instruction          | Rs1 | Rs1_Value | Rs2 | Rs2_Value | Initial_PC | Expected_PC |
          | beq x1, x2, 8        | 1   | 10        | 2   | 10        | 0x100      | 0x108       |
          | beq x1, x2, 8        | 1   | 10        | 2   | 20        | 0x100      | 0x104       |
          | bne x1, x2, 8        | 1   | 10        | 2   | 20        | 0x100      | 0x108       |

    NOW GENERATE:
    - Architecture: RV32I
    - Pipeline: {pipeline_stages}-stage
    - Include at least {req['num_tests']} test cases per instruction category
    - Test categories: Arithmetic, Logical, Memory, Branch

    Generate the complete Feature file following the exact format above.
    Output ONLY the Feature file content, no explanations.
    """
        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call LLM to generate feature content

        Args:
            prompt: Complete prompt

        Returns:
            Feature file content or None if failed
        """
        try:
            if self.llm is None:
                print("❌ No LLM available")
                return None

            # Special handling for OpenAI - use chat completions without JSON mode
            if 'OpenAI' in type(self.llm).__name__:
                result = self._call_openai_text_mode(prompt)
                return result if result else None

            # For other providers, use standard _call_api
            response = self.llm._call_api(
                prompt,
                max_tokens=4000,  # Features can be long
                system_prompt="You are a hardware verification expert specializing in BDD test generation."
            )

            if self.debug:
                print(f"\n📝 LLM Response (first 500 chars):")
                print(response[:500])
                print()

            # Clean response (remove markdown code blocks if present)
            response = self._clean_response(response)

            return response

        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _call_openai_text_mode(self, prompt: str) -> Optional[str]:
        """
        Call OpenAI without JSON mode (for feature generation)
        """
        try:
            if not hasattr(self.llm, 'client'):
                print("   ❌ No OpenAI client available")
                return None

            print(f"   📡 Calling OpenAI (text mode)...")

            messages = [
                {
                    "role": "system",
                    "content": "You are a hardware verification expert specializing in BDD test generation. Generate feature files in strict Gherkin format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Check if GPT-5 model
            model = self.llm.model

            # Build parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": 1,  # GPT-5 only supports temperature=1
            }

            # Use correct token parameter
            if 'gpt-5' in model.lower():
                params['max_completion_tokens'] = 16000
            else:
                params['max_tokens'] = 8000

            if self.debug:
                print(f"   🔍 Model: {model}")
                print(f"   🔍 Max tokens: {'max_completion_tokens' if 'gpt-5' in model.lower() else 'max_tokens'} = 8000")

            # Call API
            response = self.llm.client.chat.completions.create(**params)

            # Debug response
            if self.debug:
                print(f"   🔍 Response ID: {response.id}")
                print(f"   🔍 Model used: {response.model}")
                print(f"   🔍 Finish reason: {response.choices[0].finish_reason}")

            # Get content
            content = response.choices[0].message.content

            # Check if content is None or empty
            if content is None or len(content) == 0:
                print(f"   ⚠️  WARNING: Empty or None response from OpenAI")
                print(f"   🔍 Finish reason: {response.choices[0].finish_reason}")

                # Try to get refusal message (GPT-5 feature)
                message = response.choices[0].message
                if hasattr(message, 'refusal') and message.refusal:
                    print(f"   ❌ Content Refusal: {message.refusal}")

                # Print usage info
                if hasattr(response, 'usage') and response.usage:
                    print(f"   📊 Tokens used: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")

                return None

            if self.debug:
                print(f"   ✅ Received {len(content)} characters")
                print(f"\n📝 LLM Response (first 500 chars):")
                print(content[:500])
                print()

            # Clean response
            content = self._clean_response(content)

            return content

        except Exception as e:
            print(f"❌ OpenAI call failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None

    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response

        Remove markdown code blocks, extra whitespace, etc.
        """
        # Remove markdown code blocks
        if '```gherkin' in response:
            response = response.split('```gherkin')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]

        return response.strip()

    def _save_feature(self, content: str, req: Dict) -> Path:
        """
        Save feature file

        Args:
            content: Feature file content
            req: Requirements dict

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        module_type = req.get('module_type', 'alu')

        # Generate filename based on module type
        if module_type == 'cpu':
            pipeline = req.get('pipeline_stages', 3)
            filename = f"riscv_cpu_{pipeline}stage_{timestamp}.feature"
        elif module_type == 'regfile':
            depth = req.get('depth', 16)
            filename = f"regfile_{req['bitwidth']}bit_{depth}x_{timestamp}.feature"
        else:
            filename = f"{module_type}_{req['bitwidth']}bit_{timestamp}.feature"

        filepath = self.output_dir / filename

        # Add header with author info
        header = f"""# ==============================================================================
    # BDD Test Scenarios - Hardware Verification
    #
    # Project: LLM-based Hardware Verification Pipeline
    # Authors: Rolf Drechsler, Qian Liu
    # Paper: https://arxiv.org/abs/2512.17814
    #
    # Generated by: feature_generator_llm.py
    # LLM Provider: {self.llm_name}
    # Generated at: {timestamp}
    # Module Type: {module_type}
    # Bitwidth: {req['bitwidth']}
    # ==============================================================================

    """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)

        return filepath


def interactive_mode(project_root: Optional[str] = None):
    """
    Interactive mode - matches spec_generator style
    """
    print("=" * 70)
    print("🤖 Feature Generator - Interactive Mode (Multi-LLM Support)")
    print("=" * 70)
    print()
    print("📝 Examples of requirements:")
    print("   • '16-bit ALU'  (Simple - auto generates ADD, SUB, AND, OR)")
    print("   • '32-bit ALU with ADD, SUB, XOR'  (Medium - specify operations)")
    print("   • '16-bit ALU: ADD(0000), SUB(0001), 8 tests per op'  (Detailed)")
    print()
    print("💡 Tip: The more detailed your requirements, the better the feature!")
    print("   Type 'quit' or 'exit' to stop.")
    print()

    # Proxy status
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    if https_proxy:
        print(f"🌐 代理状态: 已设置 ({https_proxy})")

    # Ask about LLM usage
    use_llm = input("\nUse LLM provider for enhanced generation? (y/n, default: n): ").strip().lower()

    llm = None
    if HAS_LLM and use_llm in ['y', 'yes']:
        print("\nSelect LLM provider:\n")
        print("📦 FREE Providers (Recommended):")
        print("  1. Local template (no external API, zero setup)")
        print("  2. Google Gemini (FREE, 60 req/min)")
        print("  3. Groq (FREE, ultra-fast)")
        print("  4. DeepSeek (FREE, Chinese LLM) 🇨🇳")
        print()
        print("💰 PAID Providers (High Quality):")
        print("  5. OpenAI GPT-5 Series 🎯")
        print("  6. Anthropic Claude (premium)")

        choice = input("\nYour choice (1-6): ").strip()

        if choice == "2":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("\n💡 Get FREE key at: https://makersuite.google.com/app/apikey")
                api_key = input("Enter Gemini API key: ").strip()
            llm = LLMFactory.create_provider("gemini", api_key=api_key)
        elif choice == "3":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("\n💡 Get FREE key at: https://console.groq.com/keys")
                api_key = input("Enter Groq API key: ").strip()
            llm = LLMFactory.create_provider("groq", api_key=api_key)
        elif choice == "4":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("\n💡 Get key at: https://platform.deepseek.com/")
                api_key = input("Enter DeepSeek API key: ").strip()
            llm = LLMFactory.create_provider("deepseek", api_key=api_key)
        elif choice == "5":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("\n💡 Get API key at: https://platform.openai.com/api-keys")
                api_key = input("Enter OpenAI API key: ").strip()

            print("\n🤖 Select GPT-5 model:")
            print("  1. gpt-5-mini (Recommended)")
            print("  2. gpt-5")
            print("  3. gpt-5.1")
            print("  4. gpt-5.1-codex")
            model_choice = input("Model choice (1-4, default: 1): ").strip() or "1"

            model_map = {"1": "gpt-5-mini", "2": "gpt-5", "3": "gpt-5.1", "4": "gpt-5.1-codex"}
            model = model_map.get(model_choice, "gpt-5-mini")

            llm = LLMFactory.create_provider("openai", api_key=api_key, model=model)
            print(f"✅ Using OpenAI GPT-5 model: {model}")

        elif choice == "6":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("\n💡 Get API key at: https://console.anthropic.com/")
                api_key = input("Enter Anthropic API key: ").strip()
            llm = LLMFactory.create_provider("claude", api_key=api_key)
        else:
            # Local template mode
            if HAS_LLM:
                llm = LocalLLMProvider()
            else:
                llm = None
    else:
        # No LLM mode
        if HAS_LLM:
            llm = LocalLLMProvider()
        else:
            llm = None

    # Detect LLM name
    if llm:
        llm_name = type(llm).__name__.replace('Provider', '').lower()
        if llm_name == 'localllm':
            llm_name = 'local'
    else:
        llm_name = 'local'
        if HAS_LLM:
            llm = LocalLLMProvider()

    # Create generator
    generator = FeatureGeneratorLLM(
        llm_provider=llm_name,
        project_root=project_root,
        debug=True
    )

    if llm:
        generator.llm = llm
        generator.llm_name = llm_name

    print("\n" + "=" * 70)
    print("🚀 Ready! Enter your Feature requirements.\n")

    while True:
        try:
            user_input = input("\n💬 Enter your Feature requirements: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Bye!")
                break

            if not user_input:
                continue

            # Generate feature
            feature_path = generator.generate_feature(user_input)

            if feature_path:
                print(f"\n{'='*70}")
                print("📋 NEXT STEPS (Independent Generation Workflow):")
                print(f"{'='*70}")
                print(f"1. Review feature: {feature_path}")
                print(f"2. Extract metadata: python metadata_extractor.py")
                print(f"3. Generate testbench: python testbench_generator.py")
                print(f"   [Note: Steps 2 and 3 are DETERMINISTIC - no LLM required]")
                print(f"{'='*70}")
            else:
                print("\n❌ Failed to generate feature")

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Feature Generator (LLM-based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode (recommended) - no parameters needed!
  python feature_generator_llm.py
  
  # Command line - simple
  python feature_generator_llm.py -i "16-bit ALU"
  
  # Command line - medium
  python feature_generator_llm.py -i "32-bit ALU with ADD, SUB, XOR"
  
  # Command line - detailed
  python feature_generator_llm.py -i "16-bit ALU: ADD(0000), SUB(0001), 8 tests" --llm groq
        '''
    )

    parser.add_argument('-i', '--input',
                       help='User requirements in natural language')
    parser.add_argument('--llm', default='groq',
                       help='LLM provider (groq, deepseek, openai, claude, gemini)')
    parser.add_argument('--output', help='Output base directory')
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Enable debug output')

    args = parser.parse_args()

    # If no input, enter interactive mode
    if not args.input:
        interactive_mode(project_root=args.project_root)
        return 0

    # Command line mode
    generator = FeatureGeneratorLLM(
        llm_provider=args.llm,
        output_dir=args.output,
        project_root=args.project_root,
        debug=args.debug
    )

    # Generate feature
    feature_path = generator.generate_feature(args.input)

    if feature_path:
        print(f"\n🎉 Success!")
        print(f"\n📋 Next steps:")
        print(f"   1. Review: {feature_path}")
        print(f"   2. Extract metadata: python metadata_extractor.py")
        print(f"   3. Generate testbench: python testbench_generator.py")
        return 0
    else:
        print(f"\n❌ Failed to generate feature")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())