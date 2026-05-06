# rag_qwen_demo.py
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# ==================== 配置区 ====================
# 请替换为你的通义千问 API Key
DASHSCOPE_API_KEY = "sk-2ac2f01cce4947f88f7ad0184b822964"
# Milvus 连接参数（需与重建时一致）
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "finance_regulations_qwen"
# 通义千问模型选择（qwen-turbo / qwen-plus / qwen-max）
QWEN_MODEL = "qwen3-32b"
# ===============================================

# 设置 API Key
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

def main():
    print("=" * 50)
    print("金融制度 RAG 问答系统（通义千问）")
    print("=" * 50)

    # 1. 连接向量库
    print("\n[1/3] 连接 Milvus 向量库...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    print("向量库连接成功。")

    # 2. 初始化 LLM
    print(f"[2/3] 初始化通义千问模型：{QWEN_MODEL}...")
    llm = ChatTongyi(
        model=QWEN_MODEL,
        model_kwargs={"enable_thinking": False}
    )

    # 3. 构建 RAG 链
    print("[3/3] 构建 RAG 问答链...\n")

    # 提示模板（包含溯源要求）
    template = """你是一个专业的金融制度顾问。请严格根据以下提供的制度条文回答用户问题。
要求：
1. 回答必须准确、简洁，不得编造信息。
2. 在回答中注明引用的制度文件名称或条文出处。
3. 如果提供的条文中没有相关信息，请明确告知用户"当前知识库中未找到相关内容"。

【制度条文】
{context}

【用户问题】
{question}

【回答】"""

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', '未知文件')
            file_name = os.path.basename(source)
            formatted.append(f"[文件：{file_name}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("系统就绪！输入问题开始查询（输入 'exit' 退出）\n")

    # 交互式问答循环
    while True:
        query = input("请输入问题：").strip()
        if query.lower() in ['exit', 'quit', '退出']:
            print("再见！")
            break
        if not query:
            continue

        print("\n思考中...")
        try:
            answer = rag_chain.invoke(query)
            print("\n" + "=" * 50)
            print("回答：")
            print(answer)
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"\n❌ 出错：{e}\n")

if __name__ == "__main__":
    main()