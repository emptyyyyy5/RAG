# rebuild_kb_qwen.py
import os
import win32com.client
import pythoncom
from docx import Document as DocxDocument
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus

# ==================== 配置区 ====================
# 请替换为你的通义千问 API Key
DASHSCOPE_API_KEY = "sk-2ac2f01cce4947f88f7ad0184b822964"
# 文档存放的文件夹路径
DOCS_FOLDER = r"C:\Users\q1948\Desktop\project\rag\RAG"
# Milvus 连接参数
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "finance_regulations_qwen"  # 新建 collection，避免与旧版本冲突
# ===============================================

# 设置 API Key
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# ---------- Word 文档解析 ----------
def parse_word_with_win32(file_path: str) -> str:
    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(file_path)
        text = doc.Content.Text
        return text
    except Exception as e:
        raise RuntimeError(f"Word 解析失败：{file_path}，错误：{e}")
    finally:
        if doc is not None:
            try:
                doc.Close()
            except:
                pass
        if word is not None:
            try:
                word.Quit()
            except:
                pass
        pythoncom.CoUninitialize()

def parse_file(file_path: str) -> Document:
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.docx':
        doc = DocxDocument(file_path)
        text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    elif ext == '.doc':
        text = parse_word_with_win32(file_path)
    elif ext == '.txt':
        for enc in ['utf-8', 'gbk', 'gb2312', 'ansi']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError(f"无法识别文件编码：{file_path}")
    else:
        raise ValueError(f"不支持的文件类型：{ext}")

    # 新增：空内容检查
    if not text.strip():
        raise ValueError(f"文件内容为空（可能是扫描件或乱码）：{file_path}")
    
    # 新增：打印内容长度，方便排查
    print(f"   内容长度：{len(text)} 字符")
    
    return Document(
        page_content=text,
        metadata={
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "char_count": len(text)   # 存入元数据便于后续过滤
        }
    )

def load_documents_from_folder(folder_path: str) -> List[Document]:
    all_docs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.doc', '.docx', '.txt')):
                file_path = os.path.join(root, file)
                try:
                    all_docs.append(parse_file(file_path))
                    print(f"✔ 已解析：{file_path}")
                except Exception as e:
                    print(f"✘ 解析失败：{file_path}，错误：{e}")
    return all_docs

# ---------- 主流程 ----------
if __name__ == "__main__":
    print("=" * 50)
    print("开始使用通义千问 Embedding 重建向量库")
    print("=" * 50)

    # 1. 加载文档
    print("\n[1/4] 加载文档...")
    docs = load_documents_from_folder(DOCS_FOLDER)
    print(f"共加载 {len(docs)} 个文档")

    if len(docs) == 0:
        print("没有找到任何文档，请检查文件夹路径。")
        exit()

    # 2. 文本分块
    print("\n[2/4] 文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # 加大，减少条文被截断
        chunk_overlap=150,    # 加大重叠，避免跨块信息丢失
        separators=[
            "\n第", "\n一、", "\n二、", "\n三、",   # 章节边界
            "\n（一）", "\n（二）",                  # 条款边界  
            "\n\n", "\n", "。", "；", " ", ""
        ]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"共生成 {len(chunks)} 个文本块")

    # 3. 初始化千问 Embedding
    print("\n[3/4] 初始化通义千问 Embedding 模型...")
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    # 4. 存入 Milvus（会创建新 collection）
    print("\n[4/4] 向量化并存入 Milvus...")
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024}
        },
        drop_old=True   # 如果 collection 已存在则删除重建，确保维度一致
    )
    print("\n✅ 向量库重建完成！")
    print(f"Collection 名称：{COLLECTION_NAME}")