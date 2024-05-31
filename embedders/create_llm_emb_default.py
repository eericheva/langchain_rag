import os

from langchain_community.embeddings import HuggingFaceEmbeddings

from setup import Config


def create_llm_emb_default():
    llm_emb = HuggingFaceEmbeddings(
        # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface
        # .HuggingFaceEmbeddings.html
        model_name=os.path.join(Config.MODEL_SOURCE, Config.HF_EMB_MODEL),
        model_kwargs={
            # full list of parameters for this section with explanation:
            # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
            # #sentence_transformers.SentenceTransformer
            "device": "cpu"
        },
        encode_kwargs={
            # full list of parameters for this section with explanation:
            # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
            # #sentence_transformers.SentenceTransformer.encode
            "normalize_embeddings": False
        },
    )
    return llm_emb
