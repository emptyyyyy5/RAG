import os
import re
from typing import List

# 设置 Hugging Face 国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image
import easyocr
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

# ====================== 2. 基于文档结构的智能切分 ======================
def smart_document_chunking(text: str, max_chunk_size: int = 800, overlap_sentences: int = 1) -> List[str]:
    """
    基于文档结构的语义切分，尽量保持段落、标题、句子的完整性。
    
    策略：
    1. 按换行符分割为原始段落（保留自然分段）。
    2. 识别标题行（如“第一章”、“第一条”、“1.”等）作为单独块。
    3. 对每个非标题段落，若长度超过 max_chunk_size，则按中文句子边界拆分。
    4. 可选的句子级重叠：保留前一个段落的最后 overlap_sentences 个句子作为重叠。
    """
    if not text:
        return []

    # 清洗：将多个连续换行归一化为单个换行，但保留分段信息
    text = re.sub(r'\n\s*\n', '\n', text)
    lines = text.split('\n')
    
    # 1. 定义标题检测模式（可根据金融制度文档扩充）
    title_patterns = [
        r'^第[一二三四五六七八九十百千万0-9]+[章节条款条]',   # 第一章、第一条、第1条
        r'^[0-9]+[\.、]',                                 # 1. 2、
        r'^[一二三四五六七八九十]+[、]',                   # 一、
        r'^（[一二三四五六七八九十]+）',                    # （一）
        r'^[A-Z][\.、]',                                  # A.
        r'^第[0-9]+节',                                   # 第1节
        r'^[0-9]+\.[0-9]+'                                # 1.1
    ]
    def is_title(line: str) -> bool:
        line_stripped = line.strip()
        if not line_stripped:
            return False
        for pat in title_patterns:
            if re.match(pat, line_stripped):
                return True
        return False

    # 2. 句子分割（中文+英文）
    def split_sentences(para: str) -> List[str]:
        # 匹配句号、感叹号、问号、分号、省略号等作为分隔符，保留标点
        sentences = re.split(r'(?<=[。！？；…])', para)
        # 过滤空字符串，并去除首尾空格
        return [s.strip() for s in sentences if s.strip()]

    # 3. 对超长段落按句子拆分
    def split_long_paragraph(para: str, max_size: int) -> List[str]:
        if len(para) <= max_size:
            return [para]
        sentences = split_sentences(para)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_size:
                current += sent
            else:
                if current:
                    chunks.append(current.strip())
                # 如果单个句子就超过 max_size，强制截断（极少见）
                if len(sent) > max_size:
                    # 回退到按字符切分（保留最后机会）
                    for i in range(0, len(sent), max_size):
                        chunks.append(sent[i:i+max_size])
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    # 4. 主流程：逐段落处理
    raw_chunks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 标题行单独成块
        if is_title(line):
            raw_chunks.append(line)
            i += 1
            continue

        # 普通段落：收集连续的非空行（避免被换行切断的同一段落）
        paragraph_lines = []
        while i < len(lines) and lines[i].strip() and not is_title(lines[i].strip()):
            paragraph_lines.append(lines[i].strip())
            i += 1
        paragraph = " ".join(paragraph_lines)   # 段落内合并为一行
        if paragraph:
            # 按长度切分该段落
            sub_chunks = split_long_paragraph(paragraph, max_chunk_size)
            raw_chunks.extend(sub_chunks)

    # 5. 可选：添加句子级重叠（提高召回）
    if overlap_sentences > 0:
        overlapped_chunks = []
        for idx, chunk in enumerate(raw_chunks):
            if idx == 0:
                overlapped_chunks.append(chunk)
                continue
            # 从前一个chunk的末尾取 overlap_sentences 个句子
            prev_sentences = split_sentences(raw_chunks[idx-1])
            overlap_text = "".join(prev_sentences[-overlap_sentences:]) if prev_sentences else ""
            new_chunk = overlap_text + chunk
            overlapped_chunks.append(new_chunk.strip())
        return overlapped_chunks
    else:
        return raw_chunks

# 为了兼容旧代码，将原函数名替换为新实现（或保留原函数名但改逻辑）
def clean_and_split_text(text: str, chunk_size: int = 800, overlap: int = 0) -> list:
    """
    升级版切分函数，调用智能文档切分器。
    参数说明：
        chunk_size: 最大字符数（推荐600~1000）
        overlap: 重叠句子数量（默认0，若需重叠可设为1或2）
    """
    # 注意：新实现中的 overlap 参数含义是“重叠句子个数”，不是字符数
    # 如果原调用传递的是字符重叠数，这里做个近似转换（可选）
    # 简单处理：若overlap > 0，则开启句子重叠，重叠1~2句效果较好
    overlap_sentences = 1 if overlap > 0 else 0
    return smart_document_chunking(text, max_chunk_size=chunk_size, overlap_sentences=overlap_sentences)

# ====================== 3. 向量库存储（保持不变） ======================
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def save_to_milvus(chunks: list, collection_name: str = "finance_ocr"):
    Milvus.from_texts(
        texts=chunks,
        embedding=embedding,
        collection_name=collection_name,
        connection_args={"host": "localhost", "port": 19530}
    )
    print("✅ 成功！图片 OCR 结果已存入 Docker Milvus 向量库！")

# ====================== 4. 完整流程 ======================
def image_2_vector(image_path: str):
    print(f"正在处理图片: {image_path}")
    text = ocr_image_to_text(image_path)
    if not text:
        print("⚠️ OCR 未提取到任何文本")
        return
    print(f"OCR 提取文本长度: {len(text)} 字符")
    
    # 使用新的切分方法，chunk_size 建议 800，overlap 设为 1（句子级重叠）
    chunks = clean_and_split_text(text, chunk_size=800, overlap=1)
    print(f"文本已切分为 {len(chunks)} 个段落")
    
    save_to_milvus(chunks)
    print("🎉 所有步骤完成！")

if __name__ == "__main__":
    image_2_vector("test.png")