# rag-summary-and-practice
RAG 技术要点、本地实践

## 0.背景

最近几周工作上，接触些 RAG 内容，看了点资料；本着`最好的学习是复述`原则，把所有要点，重新梳理下。

思路：

1.RAG 解决什么问题？
2.RAG 核心原理、核心组件
3.RAG 高级技术，不同组件的进阶
4.效果评估
5.后续发展方向

## 1.RAG 解决什么问题

LLM 基于大规模数据的预训练，获取的通用知识。对于`私有数据`和`高频更新数据`，LLM 无法及时更新。如果采用 `Fine-Tuning` 监督微调方式，LLM 训练成本也较高，而且无法解决`幻觉`问题。 

即，`私有数据`和`高频更新数据`，以及`幻觉`问题，LLM 模型自身解决成本较高，因此，引入 RAG `Retrieval Augmented Generation`。


## 2.核心原理

RAG 检索增强生成：通过检索`外部数据源`信息，构造`融合上下文`（Context），输入给 LLM，获取更准确的结果。

核心环节：

a. 索引（indexing）
b. 检索（retrieval）
c. 生成（generation）


下述 RAG 架构图中，出了上面 3 个核心环节，还有：查询优化、路由、查询构造

* 查询优化（Query Translation）：查询重写、查询扩展、预查伪文档；
* 路由（Routing）：根据查询，判断从哪些数据源，获取信息；
* 查询抽取（Query Construction）：从原始 Query 中，抽取 SQL 、 Cypher、metadatas，分别用于 关系数据库、图数据库、向量数据库的查询。

![rag_detail_v2](../img/rag-overview.png)


开始之前，先在本地安装好 Ollama，并且下载好 embedding model 和 language model。

* TODO：增加一个链接.

安装依赖：

* TODO 增加 python 依赖以及版本？

```
! pip install langchain_community tiktoken langchain-ollama langchainhub chromadb langchain
```

### 2.1. RAG Oveview

完整的 indexing、retrieval、generation 实例代码如下：

```
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings

#### 1.INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OllamaEmbeddings(model="nomic-embed-text"))

retriever = vectorstore.as_retriever()

#### 2.RETRIEVAL and 3.GENERATION ####

# Prompt
# Pull a pre-made RAG prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

# LLM
llm = OllamaLLM(model="deepseek-r1:8b")

# Post-processing
# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to remove <think> part in the text
def remove_think_tags(text):
    """remove <think> part in the text"""
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    return cleaned_text.strip()

# RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    # | remove_think_tags
)

# Question
# Ask a question using the RAG chain
response = rag_chain.invoke("What is Task Decomposition?")
print(response)
```

### 2.2. Indexing

几个方面：

1. Tokenizer：分词，文本会被拆分成 token，映射到词表中 tokenID。
2. Embedding：嵌入，将 tokenID 映射到向量空间中，得到 token 的向量表示。
3. Chunk：分块，将文本拆分成多个 chunk，每个 chunk 包含多个 token。
4. Index：索引，将 chunk 的向量表示存储到向量数据库中。

#### 2.2.1.Token

更多细节， [Count tokens](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) and [~4 char / token](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)

> TODO: token 的扩展信息，参考上面链接.

查看下面分词得到的 Token：

```
import tiktoken

# Documents
document = "My favorite pet is a cat."
question = "What kinds of pets do I like?"

# count token num
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokenIDs = encoding.encode(string)

    print('tokenIDs: ' + str(tokenIDs))

    num_tokens = len(tokenIDs)
    return num_tokens

# use cl100k_base encoding
result = num_tokens_from_string(question, "cl100k_base")
print('token num: ' + str(result))
```

#### 2.2.2.Embedding

[Ollama Embedding](https://python.langchain.com/docs/integrations/text_embedding/ollama/) ，实例：

```
from langchain_ollama import OllamaEmbeddings

embd = OllamaEmbeddings(model="nomic-embed-text")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
result = len(query_result)

print('query_result: ' + str(query_result))
print('embedding dim: ' + str(result))
```

衡量 2 个 embedding 结果的关联关系，使用 `cosine similarity`：

```
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)
```

> TODO: 增加 cosine similarity 物理含义的说明.

#### 2.2.3.Chunk

LangChain 提供了关联工具：

* [Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)：加载各类文档数据，并转换为 LangChain 的 Document 标准对象。
* [Text Splitters](https://python.langchain.com/api_reference/text_splitters/index.html)：将文本拆分成多个 chunk，每个 chunk 包含多个 token。

下面使用 `RecursiveCharacterTextSplitter` 进行分割：

```
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Print splits
print("Print splits 1:" + splits[0])
```

> RecursiveCharacterTextSplitter: 原理细节，TODO


#### 2.2.4.Index

有多种向量数据库，下面使用 Chroma 进行演示：

```
# Index
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OllamaEmbeddings(model="nomic-embed-text"))

retriever = vectorstore.as_retriever()
```

### 2.3. Retrieval

上面建好了索引，现在进行检索：

```
# TODO: 参数含义
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("What is Task Decomposition?")

print(f"Retrieved {len(docs)} documents")
print(docs[0])
```

### 2.4. Generation

![](../img/overview-generation.png)

代码示例：

```
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = OllamaLLM(model="deepseek-r1:8b")

# Chain
chain = prompt | llm

# Run
chain.invoke({"context":docs,"question":"What is Task Decomposition?"})
```


也可以使用封装的 prompt 模板，同时，构造完整的 RAG Chain：

```
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Pull a pre-made RAG prompt from LangChain Hub
prompt_hub_rag = hub.pull("rlm/rag-prompt")

print("prompt_hub_rag: " + str(prompt_hub_rag))

# RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run
rag_chain.invoke("What is Task Decomposition?")
```











关联资料

* [rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)
* [rag-ecosystem](https://github.com/FareedKhan-dev/rag-ecosystem)