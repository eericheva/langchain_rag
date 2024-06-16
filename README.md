# RAG locally with LangChain RunnableSequence. HOWTOs

Repo contains scripts with overly detailed explanations as well as advanced scripts with not an excessive number of
details and comments (ready to run ones). These resources aim to provide someone with concise guidance and practical
examples for creating and evaluating a RAG system from scratch.

Beginners start : [`start_here.py`](tutorials/start_here.py)

Further methods : [Advanced option to rule RAG](#item-one)

### What is RAG?

~~_"Baby, don't hurt me..."_~~

RAG = Retrieval Augmented Generation
- __Retrieval__ - the process of searching for and extracting relevant information (retriever).
- __Retrieval Augmented__ - supplementing the user's query with found relevant information.
- __Retrieval Augmented Generation__ - generating a response to the user while taking into account additionally found relevant information.

**Walkthrough example:**

1) User query: _"Baby, don't hurt me..."_
2) RAG process:
   - Input Interpretation: The system receives the user's plea and detects a potential for a song lyric reference.
   - Data Retrieval: It quickly scours the attached database for relevant information, focusing on the lyrics of the song "What is Love" by Haddaway.
   - Augmentation: Next, it augments the user's query with additional context, ensuring a deep understanding of the reference.
   - Generation: Armed with the knowledge of the song's lyrics, the system crafts a witty response, perhaps something like: _"No worries, user! I'll only hurt you with my endless knowledge of 90s pop hits."_
3) RAG delivery: Finally, the system delivers the response with a touch of humor, leaving the user amused and impressed by the AI's cleverness.


### Why RAG?

- **Economically Efficient Deployment**: The development of chatbots typically starts with basic models, which are LLM models trained on generalized data. RAG offers a more cost-effective method for incorporating new data into LLM, without finetuning whole LLM.

- **Up-to-Date Information**: RAG enables to integrate rapidly changing and the latest data directly into generative models. By connecting LLM to real-time social media feeds or news websites, users receive the most current information.

- **Increased User Trust**: With RAG, LLM can provide accurate information while citing sources, boosting user confidence in the generative AI. Users can verify information by accessing the source documents themselves, enhancing trust in the system.

### How to read and create RAG:

- with __RunnableSequences__ (langchain) _(if you want clean and structured approach and easy-to-follow code sequences)_
- with __HuggingFace__ models _(if you want to try some the very resent releases and cutting-edge technology)_
- __localy__ _(if you love the smell of code in the morning)_

You can start with [`start_here.py`](tutorials/start_here.py) file and proceed with other exceptionally detailed for the begginers files and notebooks from [tutorials](tutorials) section.

### Where to find the model and how to choose one:

How to choose retrieval model (llm embedder)? --> [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard),
tab:
Retrieval or Retrieval w/Instruction

How to choose reranking model (reorder list of relevant documents)?
--> [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard), tab:
Reranking

How to choose generator model (llm for generate final answer)?
--> [open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

<a id="item-one"></a>
## Advanced option to rule RAG

Pls, refer to the other options and files listed below, to get less commented, but more advanced scripts, examples and
techniques.

<table>
    <thead>
        <tr>
            <th>HOWTO</th>
            <th>Option</th>
            <th>Go-to file</th>
            <th>Outer documentation</th>
        </tr>
    </thead>
    <tbody>
<tr>
<td colspan=4 align="center">Basic tutorials</td>
</tr>
<tr>
            <td align="left">Basic and simple</td>
            <td align="left">default</td>
            <td align="left">

[start_here.py](tutorials/start_here.py)</td>
<td align="left"></td>
</tr>
<tr>
<td colspan=4 align="center">Run scripts for full RAG system</td>
</tr>
<tr>
            <td rowspan=2 align="left">How to run HuggingFace models</td>
            <td align="left">localy:

- with HuggingFaceEmbeddings
- with HuggingFacePipeline

</td>
            <td align="left">

[local_rag_chain_simple.py](run_examples/local_rag_chain_simple.py)

[local_rag_retrieval_qa_class.py](run_examples/local_rag_retrieval_qa_class.py) </td>
<td align="left"></td>
</tr>
<tr>
<td align="left">remotely:

- with HuggingFaceHub</td>

<td align="left">

_in progress... release imminent_</td>
<td align="left">

[Hugging Face Hub documentation](https://huggingface.co/docs/hub/en/index) </td>
</tr>
        <tr>
        <td align="left">How to evaluate and monitoring application</td>
        <td align="left">with LangSmith</td>
        <td align="left">

_in progress... release imminent_</td>
        <td align="left">

[Get started with LangSmith](https://docs.smith.langchain.com/) </td>
</tr>
</tbody>
</table>

## Individual components and elements

<table>
    <thead>
        <tr>
            <th>HOWTO</th>
            <th>Option</th>
            <th>Go-to file</th>
            <th>Outer documentation</th>
        </tr>
    </thead>
    <tbody>

<tr>
<td colspan=4 align="center">How to store and embed documents?</td>
</tr>
<tr>
<td align="left">How to store embeddings in vectorstore (FAISS or Chroma)</td>
<td align="left">

default with:

- text splitter
- progress bar on creating vectorstore
- dump and load from disk </td>
  <td align="left">

[get_vectorstore.py](vectorstores/get_vectorstore.py)

[create_vectorstore.py](vectorstores/create_vectorstore.py)</td>
<td align="left">

[FAISS](https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/)

[Chroma](https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/) </td>
</tr>
<tr>
<td rowspan=3 align="left">How to embed documents</td>
<td align="left">default</td>
<td align="left">

[create_llm_emb_default.py](embedders/create_llm_emb_default.py) </td>
<td align="left">

[Text embedding models](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/) </td>
</tr>
<tr>
<td align="left">

with Caching _(save your time while next creating)_</td>
<td align="left">

[create_llm_emb_cached.py](embedders/create_llm_emb_cached.py) </td>
<td align="left">

[Caching Embeddings](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/caching_embeddings/) </td>
</tr>
<tr>
<td align="left">

with Compressing _(save RAM while store and retrieving)_</td>
<td align="left">

_in progress... release imminent_</td>
<td align="left"></td>
</tr>
<tr>
<td colspan=4 align="center">How RunnableSequence chains work?</td>
</tr>
<tr>
<td rowspan=4 align="left">How to retrieve documents</td>
<td align="left">default</td>
<td align="left">

[local_rag_chain_simple.py](run_examples/local_rag_chain_simple.py)

[combine_simple_RAG_chains.py](chains/combine_simple_RAG_chains.py) </td>
<td align="left">
</tr>
<tr>
<td align="left">with Multiple Queries Generation</td>
<td align="left">

[local_rag_chain_multi_query.py](run_examples/local_rag_chain_multi_query.py)

[multiple_queries_chain.py](chains/multiple_queries_chain.py)</td>
<td align="left"></td>
</tr>
<tr>
<td align="left">

with `chain_type` :

- `stuff`,
- `map_reduce`,
- `refine`,
- `map_rerank`</td>
  <td align="left">

  _in progress... release imminent_</td>
  <td align="left"></td>
  </tr>
  <tr>
  <td align="left">with Prompting

  Hint: ask GPT to provide instruction for your RAG system and use it as prompt template</td>
  <td align="left">

  [prompt_templates_retrieve.py](tools/prompt_templates_retrieve.py)</td>
  <td align="left"></td>
  </tr>
  <tr>
        <td rowspan=5 align="left">How to generate answer</td>
        <td align="left">default</td>
        <td align="left">

  [create_llm_gen_default.py](generators/create_llm_gen_default.py) </td>
  <td align="left"></td>

</tr>
  <tr>
        <td align="left">with Prompting

Hint: ask GPT to provide instruction for your RAG system and use it as prompt template
</td>
        <td align="left">

[prompt_templates_generate.py](tools/prompt_templates_generate.py)</td>
        <td align="left"></td>
</tr>
<tr>
        <td align="left">

with GPTQQuantizer _(save RAM and fast generation)_</td>
        <td align="left">

`pip install optimum auto-gptq`

  [create_llm_gen_default.py](generators/create_llm_gen_default.py) </td>
        <td align="left"></td>
</tr>
<tr>
        <td align="left">

with vLLM _(If you encounter `RuntimeError: probability tensor contains either inf, nan or element < 0` during `GPTQQuantizer` inference)_
</td>
        <td align="left">

`pip install vllm`

[create_llm_gen_vLLM.py](generators/create_llm_gen_vLLM.py)</td>
        <td align="left">

[vLLM in LangChain](https://python.langchain.com/v0.2/docs/integrations/llms/vllm/)
</td>
</tr>

<tr>
        <td align="left">

with LlamaCpp _(save RAM and fast generation)_
</td>
        <td align="left">

`pip install llama-cpp-python`

[create_llm_gen_llama_cpp.py](generators/create_llm_gen_llama_cpp.py)</td>
        <td align="left">

[LlamaCpp in LangChain](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/#gpu)
</td>
</tr>

</tbody>

</table>
