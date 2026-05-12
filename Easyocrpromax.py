import os
import re
from typing import List, Optional, Dict, Union, Tuple

# 设置 Hugging Face 国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image
import easyocr
import fitz  # PyMuPDF，处理 PDF 用
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

# ====================== 1. EasyOCR 配置 ======================
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

# ====================== 2. 面向金融文档的智能切分（保持不变）======================
def smart_finance_chunking(
    text: str,
    max_chunk_size: int = 500,
    overlap_sentences: int = 0,
    add_section_prefix: bool = True
) -> Union[List[str], Tuple[List[str], List[Dict]]]:
    if not text:
        return ([], []) if add_section_prefix else []

    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'(〔?\d{4}〕?\d+号)', r'\n\1\n', text)
    text = re.sub(r'(各[^：\n]*：)', r'\n\1\n', text)
    text = re.sub(r'([：:])[ \t]*', r'\1\n', text)
    lines = text.split('\n')

    title_patterns = [
        r'^关于.*(?:通知|函|公告|报告|办法|制度|指引|规定|提示|意见)$',
        r'^[^：]+[：:]$',
        r'^[（\(]?\d{4}[）\)]?\d+号$',
        r'^[（\(]?\d{4}[）\)]?.*\d+号$',
        r'^第[一二三四五六七八九十百千0-9]+[章节条款节]',
        r'^[0-9]+[\.\、\)]',
        r'^[一二三四五六七八九十]+[、\）\)]',
        r'^（[一二三四五六七八九十]+）',
        r'^[A-Z][\.\、]',
        r'^[0-9]+\.[0-9]+',
        r'^[0-9]+\.[0-9]+\.[0-9]+',
        r'^第[0-9]+节',
        r'^【[^】]+】',
        r'^[（\(]\s*[0-9]+\s*[）\)]',
        r'^条款[0-9]+',
        r'^附件[一二三四五六七八九十]*',
        r'^[0-9]+[．\.]?\s',
    ]

    def is_title(line: str) -> bool:
        line_stripped = line.strip()
        if not line_stripped:
            return False
        for pat in title_patterns:
            if re.match(pat, line_stripped):
                return True
        return False

    def split_sentences(para: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？；…])', para)
        return [s.strip() for s in sentences if s.strip()]

    def split_long_paragraph(para: str, max_size: int) -> List[str]:
        if len(para) <= max_size:
            return [para]
        sentences = split_sentences(para)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_size:
                current += sent if current == "" else sent
            else:
                if current:
                    chunks.append(current.strip())
                if len(sent) > max_size:
                    for i in range(0, len(sent), max_size):
                        chunks.append(sent[i:i+max_size])
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    raw_blocks = []
    current_section = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if is_title(line):
            current_section = line
            raw_blocks.append(('title', line))
            i += 1
            continue
        paragraph_lines = []
        while i < len(lines) and lines[i].strip() and not is_title(lines[i].strip()):
            paragraph_lines.append(lines[i].strip())
            i += 1
        paragraph = " ".join(paragraph_lines)
        if paragraph:
            raw_blocks.append(('para', paragraph))

    chunks = []
    metadata_list = []
    for block_type, content in raw_blocks:
        if block_type == 'title':
            chunks.append(content)
            metadata_list.append({'section': current_section})
        else:
            sub_chunks = split_long_paragraph(content, max_chunk_size)
            for sub in sub_chunks:
                if add_section_prefix and current_section:
                    prefixed = f"【{current_section}】 {sub}"
                else:
                    prefixed = sub
                chunks.append(prefixed)
                metadata_list.append({'section': current_section})

    if overlap_sentences > 0:
        if add_section_prefix:
            print("⚠️ 警告: 重叠功能与章节前缀冲突，已忽略重叠，返回前缀版本。")
            return chunks, metadata_list
        else:
            pure_chunks = []
            for block_type, content in raw_blocks:
                if block_type == 'title':
                    pure_chunks.append(content)
                else:
                    pure_chunks.extend(split_long_paragraph(content, max_chunk_size))
            overlapped = []
            for idx, chunk in enumerate(pure_chunks):
                if idx == 0:
                    overlapped.append(chunk)
                    continue
                prev_sentences = split_sentences(pure_chunks[idx-1])
                overlap_text = "".join(prev_sentences[-overlap_sentences:]) if prev_sentences else ""
                overlapped.append(overlap_text + chunk)
            return overlapped
    return (chunks, metadata_list) if add_section_prefix else chunks


def clean_and_split_text(text: str, chunk_size: int = 500, overlap: int = 0) -> list:
    add_prefix = True
    result = smart_finance_chunking(text, max_chunk_size=chunk_size,
                                    overlap_sentences=1 if overlap > 0 else 0,
                                    add_section_prefix=add_prefix)
    if add_prefix:
        chunks, _ = result
        return chunks
    return result

# ====================== 3. 向量库存储 ======================
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def save_to_milvus(texts: list, metadatas: Optional[List[Dict]] = None,
                   collection_name: str = "finance_ocr"):
    if metadatas:
        Milvus.from_texts(texts=texts, embedding=embedding,
                          metadatas=metadatas, collection_name=collection_name,
                          connection_args={"host": "localhost", "port": 19530})
    else:
        Milvus.from_texts(texts=texts, embedding=embedding,
                          collection_name=collection_name,
                          connection_args={"host": "localhost", "port": 19530})
    print("✅ 成功！文本块已存入 Docker Milvus 向量库！")

# ====================== 4. 批量处理入口 ======================
def process_folder(folder_path: str):
    """处理整个文件夹的图片和 PDF，统一切分并存入 Milvus"""
    if not os.path.isdir(folder_path):
        print(f"❌ 目录不存在: {folder_path}")
        return
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.pdf')
    all_text_parts = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(supported_ext):
            continue
        file_path = os.path.join(folder_path, filename)
        print(f"📄 正在处理: {file_path}")
        if filename.lower().endswith('.pdf'):
            text = pdf_to_text(file_path)
        else:
            text = ocr_image_to_text(file_path)
        if text:
            all_text_parts.append(text)
            print(f"   -> 提取 {len(text)} 字符")
        else:
            print(f"   ⚠️ 未提取到文字")
    if not all_text_parts:
        print("❌ 文件夹内没有可提取的文字，退出。")
        return
    full_text = "\n\n".join(all_text_parts)
    print(f"\n📊 合并文本总长度: {len(full_text)} 字符")
    chunks, metadatas = smart_finance_chunking(full_text, max_chunk_size=500,
                                               overlap_sentences=0, add_section_prefix=True)
    print(f"✂️  切分为 {len(chunks)} 个块")
    save_to_milvus(chunks, metadatas=metadatas, collection_name="finance_ocr")
    print("🎉 全部文件处理完成！")

if __name__ == "__main__":
    # 用法：将 test_folder 替换为你的图片/PDF 文件夹路径
    process_folder("ImgorPDF")