"""
BDD Specification Generator (Specification-First Workflow)
==========================================================

This module generates BDD Feature files from STRUCTURED parameters
(module_type, bitwidth, operations, etc.) instead of relying on
natural language regex parsing.

Used by the "Specification-First" workflow where BDD specs are
written BEFORE hardware implementation.

DIFFERENCE FROM feature_generator_llm.py:
- feature_generator_llm.py: accepts natural language → regex parse → prompt
- bdd_spec_generator.py: accepts structured params → direct prompt (no regex)

Both produce the same output: .feature files in output/bdd/{llm}/

USAGE:
    generator = BddSpecGenerator(llm_provider='groq', project_root='/path/to/project')
    filepath = generator.generate(
        module_type='alu',
        bitwidth=16,
        operations=['ADD', 'SUB', 'AND', 'OR'],
        num_tests=5,
        custom_requirements='Test overflow detection'
    )
"""

from __future__ import annotations

from typing import Dict, Optional, List

# Reuse the existing FeatureGeneratorLLM for prompt creation & file saving
try:
    from feature_generator_llm import FeatureGeneratorLLM

    HAS_BASE_GENERATOR = True
except ImportError:
    HAS_BASE_GENERATOR = False
    print("⚠️  feature_generator_llm not found, BDD Spec Generator unavailable")


class BddSpecGenerator:
    """
    Generate BDD Feature files from structured parameters.

    This is the backend for the Specification-First workflow.
    It wraps FeatureGeneratorLLM but bypasses the NL regex parsing,
    constructing the requirements dict directly from structured inputs.
    """

    # Same defaults as FeatureGeneratorLLM for consistency
    DEFAULT_OPCODES = {
        'ADD': '0000', 'SUB': '0001', 'AND': '0010', 'OR': '0011',
        'XOR': '0100', 'NOT': '0101', 'SHL': '0110', 'SHR': '0111'
    }

    DEFAULT_OPERATIONS = {
        'alu': ['ADD', 'SUB', 'AND', 'OR'],
        'counter': ['UP', 'DOWN', 'UPDOWN'],
        'regfile': ['READ', 'WRITE'],
        'cpu': ['ADD', 'SUB', 'AND', 'OR', 'XOR']
    }

    def __init__(
            self,
            llm_provider: str = 'groq',
            project_root: Optional[str] = None,
            debug: bool = True
    ):
        """
        Initialize BDD Spec Generator.

        Args:
            llm_provider: LLM provider name
            project_root: Project root directory
            debug: Enable debug output
        """
        if not HAS_BASE_GENERATOR:
            raise ImportError("feature_generator_llm is required")

        self.debug = debug
        self.llm_provider = llm_provider

        # Create the base generator (reuse its LLM setup, prompts, saving)
        self._generator = FeatureGeneratorLLM(
            llm_provider=llm_provider,
            project_root=project_root,
            debug=debug
        )

        print(f"📋 BDD Spec Generator (Specification-First)")
        print(f"   LLM: {self._generator.llm_name}")

    def build_requirements(
            self,
            module_type: str = 'alu',
            bitwidth: int = 16,
            operations: Optional[List[str]] = None,
            num_tests: int = 5,
            depth: int = 32,
            pipeline_stages: int = 5,
            custom_requirements: str = ''
    ) -> Dict:
        """
        Build a requirements dict from structured parameters.
        This replaces the NL regex parsing in parse_user_input().

        Args:
            module_type: 'alu', 'counter', 'regfile', 'cpu'
            bitwidth: 8, 16, 32, 64
            operations: list of operation names (e.g. ['ADD', 'SUB'])
            num_tests: test cases per operation (5-20)
            depth: register count for regfile
            pipeline_stages: for cpu (3 or 5)
            custom_requirements: extra NL text from user

        Returns:
            Dict matching the format of FeatureGeneratorLLM.parse_user_input()
        """
        # Validate module_type
        if module_type not in ('alu', 'counter', 'regfile', 'cpu'):
            print(f"⚠️  Unknown module_type '{module_type}', defaulting to 'alu'")
            module_type = 'alu'

        # Validate bitwidth
        if bitwidth not in (8, 16, 32, 64):
            print(f"⚠️  Invalid bitwidth {bitwidth}, defaulting to 16")
            bitwidth = 16

        # Validate num_tests
        num_tests = max(5, min(num_tests, 20))

        # Use default operations if none specified
        if not operations:
            operations = self.DEFAULT_OPERATIONS.get(module_type, ['ADD', 'SUB', 'AND', 'OR'])

        # Build opcodes from defaults
        opcodes = {}
        for op in operations:
            opcodes[op] = self.DEFAULT_OPCODES.get(op, '0000')

        # Build the original_input string (for logging / file header)
        original_input = f"{bitwidth}-bit {module_type.upper()} with {', '.join(operations)}"
        if custom_requirements:
            original_input += f" | {custom_requirements}"

        return {
            'bitwidth': bitwidth,
            'module_type': module_type,
            'operations': operations,
            'opcodes': opcodes,
            'num_tests': num_tests,
            'original_input': original_input,
            'pipeline_stages': pipeline_stages,
            'depth': depth,
            'custom_requirements': custom_requirements
        }

    def generate(
            self,
            module_type: str = 'alu',
            bitwidth: int = 16,
            operations: Optional[List[str]] = None,
            num_tests: int = 5,
            depth: int = 32,
            pipeline_stages: int = 5,
            custom_requirements: str = ''
    ) -> Optional[str]:
        """
        Generate BDD Feature file from structured parameters.

        Returns:
            Path to generated .feature file, or None on failure
        """
        # Build requirements dict (no NL regex parsing!)
        requirements = self.build_requirements(
            module_type=module_type,
            bitwidth=bitwidth,
            operations=operations,
            num_tests=num_tests,
            depth=depth,
            pipeline_stages=pipeline_stages,
            custom_requirements=custom_requirements
        )

        print(f"\n{'=' * 70}")
        print(f"📋 BDD-First: Generating spec for {bitwidth}-bit {module_type.upper()}")
        print(f"{'=' * 70}")
        print(f"   Operations: {', '.join(requirements['operations'])}")
        print(f"   Tests/op: {requirements['num_tests']}")
        if custom_requirements:
            print(f"   Custom: {custom_requirements}")

        # Create prompt using FeatureGeneratorLLM's prompt builders
        prompt = self._generator._create_prompt(requirements)

        # Append custom requirements to prompt if provided
        if custom_requirements:
            prompt += f"""

ADDITIONAL REQUIREMENTS FROM USER:
{custom_requirements}

Incorporate these additional requirements into the test scenarios.
"""

        if self.debug:
            print(f"\n📝 Prompt preview:")
            print(prompt[:500] + "...")

        # Call LLM
        print(f"\n🤖 Calling {self._generator.llm_name} API...")
        feature_content = self._generator._call_llm(prompt)

        if not feature_content:
            print("❌ Failed to generate feature")
            return None

        # Save
        feature_path = self._generator._save_feature(feature_content, requirements)

        print(f"\n✅ BDD Spec generated: {feature_path}")
        return str(feature_path)

    def generate_streaming(
            self,
            module_type: str = 'alu',
            bitwidth: int = 16,
            operations: Optional[List[str]] = None,
            num_tests: int = 5,
            depth: int = 32,
            pipeline_stages: int = 5,
            custom_requirements: str = ''
    ):
        """
        Generate BDD Feature file with streaming support.
        Yields (prompt, requirements) for the caller to handle streaming.

        Returns:
            Tuple of (prompt: str, requirements: dict)
        """
        requirements = self.build_requirements(
            module_type=module_type,
            bitwidth=bitwidth,
            operations=operations,
            num_tests=num_tests,
            depth=depth,
            pipeline_stages=pipeline_stages,
            custom_requirements=custom_requirements
        )

        prompt = self._generator._create_prompt(requirements)

        if custom_requirements:
            prompt += f"""

ADDITIONAL REQUIREMENTS FROM USER:
{custom_requirements}

Incorporate these additional requirements into the test scenarios.
"""

        return prompt, requirements

    @property
    def llm(self):
        """Access the underlying LLM instance"""
        return self._generator.llm

    @llm.setter
    def llm(self, value):
        """Set the underlying LLM instance (for model override)"""
        self._generator.llm = value

    @property
    def llm_name(self):
        return self._generator.llm_name

    def clean_response(self, response: str) -> str:
        """Delegate to base generator"""
        return self._generator._clean_response(response)

    def save_feature(self, content: str, requirements: dict) -> str:
        """Delegate to base generator"""
        return str(self._generator._save_feature(content, requirements))
