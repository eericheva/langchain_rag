import os

from langchain_community.llms import VLLM

from setup import Config


def create_llm_gen_vLLM():
    llm_gen = VLLM(
        # https://python.langchain.com/v0.2/docs/integrations/llms/vllm/
        # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.vllm.VLLM.html
        model=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=1024,
        top_p=0.9,
        temperature=0.3,
        n_ctx=-1,  # -1 stands for using original model context lenght
        task="text-generation",
        device=Config.DEVICE_GEN,  # -1 stands for CPU
        verbose=True,
        vllm_kwargs={"quantization": "gptq"},  # could be "awq", "gptq", "ft8" etc.
    )
    return llm_gen
