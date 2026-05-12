from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def search(query="异地贷款"):
    db = Milvus(
        embedding_function=embedding,
        collection_name="finance_ocr",
        connection_args={"host": "localhost", "port": 19530}
    )
    docs = db.similarity_search(query, k=3)
    print("\n🔍 检索结果：")
    for doc in docs:
        print("-", doc.page_content)

if __name__ == "__main__":
    search()