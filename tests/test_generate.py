import torch

from generate import generate
from llm_architecture import GPTModel


class DummyTokenizer:
    def encode(self, text):
        return [11, 12, 13]

    def decode(self, token_ids):
        return "prompt"


def test_generate_can_use_random_model(monkeypatch):
    monkeypatch.setattr("generate.os.path.exists", lambda _path: False)
    monkeypatch.setattr("generate.tiktoken.get_encoding", lambda _name: DummyTokenizer())

    def fake_generate_text_simple(model, idx, max_new_tokens, context_size):
        assert isinstance(model, GPTModel)
        return idx

    monkeypatch.setattr("generate.generate_text_simple", fake_generate_text_simple)

    result = generate("hello", source="random", model_path="missing.pth", max_new_tokens=3)

    assert result == "prompt"
