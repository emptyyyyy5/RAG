import os
import warnings
warnings.filterwarnings("ignore")

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["GLOG_minloglevel"] = "2"

from paddleocr import PaddleOCR

# 兼容 LangChain Document 导入
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import Milvus
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ====================== 1. PaddleOCR 初始化（尝试强制使用 CPU 和旧版模型） ======================
# 注意：新版 PaddleOCR 可能不支持 use_gpu 参数，但我们可以通过环境变量或模型选择来影响推理行为
ocr_engine = PaddleOCR(
    lang='ch',
    ocr_version='PP-OCRv4',  # 使用更稳定的 v4 模型，避免 v5 可能的不稳定
    use_textline_orientation=False,  # 关闭文本行方向分类，减少计算复杂度
    # 尝试强制使用 CPU 推理（环境变量已设）
)

def ocr_image_to_documents(image_path: str) -> list:
    """尝试多种调用方式以绕过 ONNX 错误"""
    docs = []
    
    # ---- 方式1：优先尝试 predict 方法 ----
    try:
        result = ocr_engine.predict(image_path)
        if result and len(result) > 0:
            pred = result[0]
            rec_texts = pred.get("rec_texts", [])
            rec_scores = pred.get("rec_scores", [])
            for text, conf in zip(rec_texts, rec_scores):
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": image_path, "confidence": round(conf, 4)}
                        )
                    )
            if docs:
                return docs
    except Exception as e:
        print(f"⚠️ predict 调用失败，尝试回退到 ocr 方法: {e}")
    
    # ---- 方式2：回退到旧版 ocr 方法（如果 predict 失败）----
    try:
        # 旧版 API 在 3.x 中可能依然存在
        result = ocr_engine.ocr(image_path, cls=False)  # 关闭方向分类，减少错误
        if result and result[0]:
            for line in result[0]:
                # 注意：ocr 方法返回的结构可能略有不同
                if len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                    confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                else:
                    continue
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": image_path, "confidence": round(confidence, 4)}
                        )
                    )
        if docs:
            return docs
    except Exception as e:
        print(f"⚠️ ocr 方法也失败了: {e}")
    
    # ---- 方式3：最后的尝试，使用更底层的检测+识别分开调用 ----
    try:
        # 直接使用检测模型和识别模型，绕过高层次封装
        from paddleocr import PaddleOCR
        # 获取原始模型
        det_model = ocr_engine.text_detector
        rec_model = ocr_engine.text_recognizer
        
        # 用 PIL 读取图片
        from PIL import Image
        import numpy as np
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # 检测文本区域
        dt_boxes, _ = det_model(img_np)
        if dt_boxes is not None and len(dt_boxes) > 0:
            for box in dt_boxes:
                # 裁剪图片区域进行识别
                # 此处简化处理，具体裁剪逻辑较复杂，略
                pass
    except Exception as e:
        print(f"⚠️ 底层调用也失败: {e}")
    
    print("❌ 所有识别方式均失败，请检查环境。")
    return []

# ====================== 2. 文本分段 ======================
def clean_and_split_text(text: str, chunk_size=500, overlap=50) -> list:
    text = text.replace("\n", " ").replace("  ", " ").strip()
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# ====================== 3. 嵌入模型 & Milvus 存储 ======================
embedding = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

def save_to_milvus(chunks: list):
    if not chunks:
        print("⚠️ 没有文本块可存储，请检查输入。")
        return
    try:
        Milvus.from_texts(
            texts=chunks,
            embedding=embedding,
            collection_name="finance_ocr",
            connection_args={"host": "localhost", "port": 19530}
        )
        print("✅ 成功！图片 OCR 文本已存入 Docker Milvus 向量库！")
    except Exception as e:
        print(f"❌ 存入 Milvus 失败: {e}")

# ====================== 4. 主流程 ======================
def image_2_vector(image_path: str):
    print(f"🖼️ 正在处理图片: {image_path}")
    docs = ocr_image_to_documents(image_path)
    if not docs:
        print("❌ 未识别到任何文本，流程终止。")
        return

    full_text = "\n".join([doc.page_content for doc in docs])
    print(f"📝 识别到 {len(docs)} 行文本，总字符数: {len(full_text)}")

    chunks = clean_and_split_text(full_text)
    save_to_milvus(chunks)

if __name__ == "__main__":
    image_2_vector("test.png")