import os

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from setup import Config


def create_llm_gen_llama_cpp():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm_gen = LlamaCpp(
        model_path=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9,
        callback_manager=callback_manager,
        n_gpu_layers=-1 if Config.DEVICE_GEN > -1 else 0,
        # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers
        # there are, you can use -1 to move all
        # n_batch=512,
        n_ctx=-1,  # -1 stands for using original model context lenght
        verbose=True,  # Verbose is required to pass to the callback manager
    )
    return llm_gen
