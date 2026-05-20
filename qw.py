import os
import re
from typing import List, Optional, Dict, Union, Tuple
import pythoncom
import win32com.client

# 设置 DashScope API Key 环境变量（对齐第二个脚本）
os.environ["DASHSCOPE_API_KEY"] = "sk-4e9b928c5d1848ed808e27565468f3ae"

from PIL import Image
import easyocr
import fitz  # PyMuPDF，处理 PDF 用
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus

# ====================== 1. EasyOCR 配置（保留原有 OCR 功能） ======================
reader = easyocr.Reader(['ch_sim', 'en'])

def ocr_image_to_text(image_path: str) -> str:
    """使用 EasyOCR 从图片中提取文字"""
    try:
        results = reader.readtext(image_path, detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        print(f"EasyOCR 识别失败: {e}")
        return ""

def pdf_to_text(pdf_path: str) -> str:
    """将 PDF 每一页转换为图片后 OCR，返回合并文本"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"打开 PDF 失败: {e}")
        return ""
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 渲染为图片，dpi 200 兼顾速度与清晰度
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_path = f"_temp_page_{page_num}.png"
        img.save(temp_path)
        text = ocr_image_to_text(temp_path)
        if text:
            all_text.append(text)
        os.remove(temp_path)  # 清理临时图片
    doc.close()
    return "\n".join(all_text)

# ====================== 2. 文本切分（替换为第二个脚本的 RecursiveCharacterTextSplitter） ======================
def clean_and_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    对齐第二个脚本的文本切分逻辑，返回 Document 列表
    """
    # 构建 Document 对象（对齐第二个脚本的元数据格式）
    doc = Document(
        page_content=text,
        metadata={"source": "ocr_text", "file_name": "ocr_combined.txt"}
    )
    
    # 使用第二个脚本的 RecursiveCharacterTextSplitter 切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return chunks

# ====================== 3. 向量库存储（替换为第二个脚本的 DashScope + Milvus 逻辑） ======================
# Milvus 连接参数（完全对齐第二个脚本）
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "finance_regulations_qwen"  # 与第二个脚本的 collection 名称一致

def save_to_milvus(documents: List[Document]):
    """
    对齐第二个脚本的向量存储逻辑：使用通义千问 Embedding + Milvus 重建逻辑
    """
    # 初始化千问 Embedding（完全对齐第二个脚本）
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 存入 Milvus（会创建新 collection，删除旧的）
    vector_store = Milvus.from_documents(
        documents=documents,
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
    print("✅ 成功！文本块已存入 Milvus 向量库（通义千问 Embedding）！")

# ====================== 4. 批量处理入口（整合 OCR + 新切分 + 新存储） ======================
def process_folder(folder_path: str):
    """处理整个文件夹的图片和 PDF，使用第二个脚本的切分/Embedding/存储逻辑"""
    if not os.path.isdir(folder_path):
        print(f"❌ 目录不存在: {folder_path}")
        return
    
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.pdf')
    all_text = ""
    processed_files = 0
    
    print("=" * 50)
    print("开始处理 OCR 文档并使用通义千问 Embedding 构建向量库")
    print("=" * 50)

    # 1. 批量 OCR 提取文本
    print("\n[1/3] 提取图片/PDF 文字...")
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(supported_ext):
            continue
        
        file_path = os.path.join(folder_path, filename)
        print(f"📄 正在处理: {file_path}")
        
        try:
            if filename.lower().endswith('.pdf'):
                text = pdf_to_text(file_path)
            else:
                text = ocr_image_to_text(file_path)
            
            if text:
                all_text += text + "\n\n"
                processed_files += 1
                print(f"   -> 提取 {len(text)} 字符")
            else:
                print(f"   ⚠️ 未提取到文字")
        except Exception as e:
            print(f"   ✘ 处理失败：{e}")
    
    if processed_files == 0:
        print("❌ 文件夹内没有可提取的文字，退出。")
        return
    print(f"\n📊 合并文本总长度: {len(all_text)} 字符")

    # 2. 文本分块（使用第二个脚本的切分逻辑）
    print("\n[2/3] 文本分块...")
    chunks = clean_and_split_text(all_text, chunk_size=500, chunk_overlap=50)
    print(f"共生成 {len(chunks)} 个文本块")

    # 3. 存入 Milvus（使用第二个脚本的 Embedding 和存储逻辑）
    print("\n[3/3] 向量化并存入 Milvus...")
    save_to_milvus(chunks)
    
    print("\n🎉 全部文件处理完成！")
    print(f"Collection 名称：{COLLECTION_NAME}")

if __name__ == "__main__":
    # 用法：将 ImgorPDF 替换为你的图片/PDF 文件夹路径
    process_folder("ImgorPDF")