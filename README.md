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

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)


开始之前，先在本地安装好 Ollama，并且下载好 embedding model 和 language model。

* TODO：增加一个链接.

安装依赖：

* TODO 增加 python 依赖以及版本？

```
! pip install langchain_community tiktoken langchain-ollama langchainhub chromadb langchain
```









关联资料

* [rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)
* [rag-ecosystem](https://github.com/FareedKhan-dev/rag-ecosystem)