"""GPT-5 calls ask for low reasoning, and every GPT-5 version is treated alike (D49).

Before, the calls used OpenAI's default reasoning ("medium"): 60-170 s per Step-1 or
Step-2 call. And a fixed list of model names sent gpt-5.4-mini and later versions
the old token parameter, which they reject.
"""
import pytest

lp = pytest.importorskip('src.llm_providers')


def _provider(model):
    p = object.__new__(lp.OpenAIProvider)          # no API key needed for the parameters
    p.model = model
    return p


@pytest.mark.parametrize('model', ['gpt-5-mini', 'gpt-5.4-mini', 'gpt-5.5', 'gpt-5.1'])
def test_every_gpt5_version_gets_the_gpt5_parameters_and_low_reasoning(model):
    p = _provider(model)
    assert p._is_gpt5_model(model)
    assert p._reasoning() == {'reasoning_effort': 'low'}


def test_codex_and_older_models_are_left_alone():
    assert _provider('gpt-5.3-codex')._reasoning() == {}
    assert not _provider('gpt-4o')._is_gpt5_model('gpt-4o')
