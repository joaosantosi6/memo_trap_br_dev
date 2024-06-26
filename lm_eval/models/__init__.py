from . import chatgpt
from . import gpt2
from . import together
from . import gpt3
from . import dummy

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "chatgpt": chatgpt.CHATGPTLM,
    "dummy": dummy.DummyLM,
    "together": together.TogetherLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
