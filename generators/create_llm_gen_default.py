import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from setup import Config


def create_llm_gen_default():
    llm_gen = HuggingFacePipeline.from_model_id(
        # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_pipeline
        # .HuggingFacePipeline.html
        model_id=os.path.join(Config.MODEL_SOURCE, Config.HF_LLM_NAME),
        task="text-generation",
        device=-1,  # -1 stands for CPU
        pipeline_kwargs={
            # full list of parameters for this section with explanation:
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            # Note: some of them (depends on the specific model) should go to the model_kwargs attribute
            "max_new_tokens": 512,  # How long could be generated answer
            "return_full_text": False,
            # "return_full_text": True if you want to return within generation answer also all prompts,
            # contexts and other serving instrumentals
        },
        model_kwargs={
            # full list of parameters for this section with explanation:
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            # Note: some of them (depends on the specific model) should go to the pipeline_kwargs attribute
            "do_sample": True,
            "top_k": 10,
            "temperature": 0.0,
            "repetition_penalty": 1.03,  # 1.0 means no penalty
            "max_length": 20,
        },
    )
    return llm_gen
