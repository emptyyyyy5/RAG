import os
import re
from typing import List, Optional, Dict, Union, Tuple

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

# ====================== 2. 面向金融文档的智能切分 ======================
def smart_finance_chunking(
    text: str,
    max_chunk_size: int = 500,
    overlap_sentences: int = 0,
    add_section_prefix: bool = True
) -> Union[List[str], Tuple[List[str], List[Dict]]]:
    """
    金融文档专用切分器：
    - 识别公文标题、文号、称呼、章节编号等作为独立块
    - 保持段落完整性，超长段落按句子边界拆分
    - 可选择添加章节前缀（如【一、风险认识】），增强上下文
    - 可选句子级重叠（仅在没有章节前缀时生效，避免格式混乱）

    返回：
        若 add_section_prefix=True，返回 (chunks: List[str], metadatas: List[Dict])
        否则返回 List[str]
    """
    if not text:
        return ([], []) if add_section_prefix else []

    # ---- 预处理：规范化换行，便于识别标题 ----
    text = re.sub(r'\n\s*\n', '\n', text)
    # 尝试在公文文号、称呼后强制换行，防止 OCR 丢失换行
    text = re.sub(r'(〔?\d{4}〕?\d+号)', r'\n\1\n', text)
    text = re.sub(r'(各[^：\n]*：)', r'\n\1\n', text)
    text = re.sub(r'([：:])[ \t]*', r'\1\n', text)  # 称呼冒号后换行
    lines = text.split('\n')

    # ---- 扩充后的标题检测模式 ----
    title_patterns = [
        r'^关于.*(?:通知|函|公告|报告|办法|制度|指引|规定|提示|意见)$',  # 公文标题
        r'^[^：]+[：:]$',                                  # 抬头称呼：各银监分局，...
        r'^[（\(]?\d{4}[）\)]?\d+号$',                    # 文号：苏银监办〔2005〕235号
        r'^[（\(]?\d{4}[）\)]?.*\d+号$',                  # 更宽泛的文号
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
        """按中文句末标点切分句子"""
        sentences = re.split(r'(?<=[。！？；…])', para)
        return [s.strip() for s in sentences if s.strip()]

    def split_long_paragraph(para: str, max_size: int) -> List[str]:
        """将超长段落按句子边界拆分为多个 chunk"""
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
                # 极长单句，强制按字符拆分
                if len(sent) > max_size:
                    for i in range(0, len(sent), max_size):
                        chunks.append(sent[i:i+max_size])
                    current = ""
                else:
                    current = sent
        if current:
            chunks.append(current.strip())
        return chunks

    # ---- 第一遍扫描：识别块类型和章节标题 ----
    raw_blocks = []          # 元素：('title', 内容) 或 ('para', 内容)
    current_section = ""     # 最近的章节标题（用于前缀）
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

        # 普通段落：合并连续非空且非标题的行
        paragraph_lines = []
        while i < len(lines) and lines[i].strip() and not is_title(lines[i].strip()):
            paragraph_lines.append(lines[i].strip())
            i += 1
        paragraph = " ".join(paragraph_lines)
        if paragraph:
            raw_blocks.append(('para', paragraph))

    # ---- 第二遍：生成最终 chunk 和元数据 ----
    chunks = []
    metadata_list = []

    for block_type, content in raw_blocks:
        if block_type == 'title':
            # 标题直接作为一个 chunk（利于检索）
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

    # ---- 重叠处理（仅在无前缀时有效） ----
    if overlap_sentences > 0:
        if add_section_prefix:
            print("⚠️ 警告: 重叠功能与章节前缀冲突，已忽略重叠，返回前缀版本。")
            return chunks, metadata_list
        else:
            # 生成无前缀的纯文本块用于添加句子重叠
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
            return overlapped  # 无前缀时返列表

    return (chunks, metadata_list) if add_section_prefix else chunks


# 兼容旧接口
def clean_and_split_text(text: str, chunk_size: int = 500, overlap: int = 0) -> list:
    """
    旧版兼容接口，内部调用 smart_finance_chunking。
    返回带前缀的文本块列表（忽略元数据）。
    """
    add_prefix = True
    result = smart_finance_chunking(
        text,
        max_chunk_size=chunk_size,
        overlap_sentences=1 if overlap > 0 else 0,
        add_section_prefix=add_prefix
    )
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

def save_to_milvus(
    texts: list,
    metadatas: Optional[List[Dict]] = None,
    collection_name: str = "finance_ocr"
):
    """将文本块存入 Milvus 向量库，可选附带元数据（章节信息等）"""
    if metadatas:
        Milvus.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            collection_name=collection_name,
            connection_args={"host": "localhost", "port": 19530}
        )
    else:
        Milvus.from_texts(
            texts=texts,
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

    # 使用金融文档专用切分
    chunks, metadatas = smart_finance_chunking(
        text,
        max_chunk_size=500,
        overlap_sentences=0,       # 推荐关闭重叠，用章节前缀保留上下文
        add_section_prefix=True
    )
    print(f"文本已切分为 {len(chunks)} 个段落")
    if metadatas:
        print(f"附带元数据样本: {metadatas[0]}")

    save_to_milvus(chunks, metadatas=metadatas, collection_name="finance_ocr")
    print("🎉 所有步骤完成！")


if __name__ == "__main__":
    # 示例用法：python Easyocrpro.py
    image_2_vector("test.png")