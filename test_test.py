# test_retrieval.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# 加载与构建时相同的 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="D:/desktop/pycharm/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 连接已有的 Milvus collection
vector_store = Milvus(
    embedding_function=embeddings,
    collection_name="finance_regulations",
    connection_args={"host": "localhost", "port": "19530"}
)

# 输入测试查询
query = "公安保险"
print(f"查询内容：{query}\n")

# 检索最相关的 3 个文本块
results = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"--- 结果 {i+1} ---")
    print(f"来源文件：{doc.metadata.get('source', '未知')}")
    print(f"内容片段：{doc.page_content[:200]}...")
    print()