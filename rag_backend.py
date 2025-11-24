import os
import json
import hashlib
import time
import requests
import numpy as np
import faiss
from tqdm import tqdm
from pypdf import PdfReader

# =============================
# 基础配置
# =============================

SILICONFLOW_API_KEY = "sk-nxcwqkxhmpvfahoheegvcdvrqsslbmbrluoqpxglcnicrxrc"
BASE_URL = "https://api.siliconflow.cn/v1"    # 强制使用 cn 域名

EMBEDDING_MODEL = "BAAI/bge-m3"
CHAT_MODEL = "Qwen/Qwen3-8B"

DOC_PATH = "Kindle_User's_Guide_English.pdf"
INDEX_DIR = "./rag_index"

assert SILICONFLOW_API_KEY.startswith("sk-")

os.makedirs(INDEX_DIR, exist_ok=True)

# --- Requests 配置，关闭 SSL verify（避免 handshake error）
REQUEST_TIMEOUT = 30
RETRY_TIMES = 3
SLEEP_BETWEEN_RETRY = 1.0

_REQ_KW = {
    "timeout": REQUEST_TIMEOUT,
    "verify": False,      # ⭐ 不验证 SSL（解决 SSLEOFError）
    "proxies": {}         # ⭐ 禁用系统代理
}

# 关闭 SSL 警告
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 创建全局 session 以复用连接
_session = requests.Session()
_session.verify = False
_session.proxies = {}


# =============================
# 基础请求封装
# =============================

def _headers():
    return {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }


# --- ⭐ 增强版本：添加重试机制和错误处理
def _post_json(url, payload):
    last_error = None
    
    for attempt in range(RETRY_TIMES):
        try:
            resp = _session.post(
                url, 
                headers=_headers(), 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            # 确保响应使用 UTF-8 编码
            resp.encoding = 'utf-8'
            
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except:
                    detail = resp.text
                raise RuntimeError(f"{resp.status_code} {resp.reason}: {detail}")
            
            # 确保 JSON 解析时使用 UTF-8
            result = resp.json()
            return result
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            last_error = e
            if attempt < RETRY_TIMES - 1:
                time.sleep(SLEEP_BETWEEN_RETRY * (attempt + 1))  # 指数退避
                continue
            else:
                raise RuntimeError(f"请求失败（已重试 {RETRY_TIMES} 次）: {e}") from e
        except Exception as e:
            raise RuntimeError(f"未知错误: {e}") from e
    
    raise RuntimeError(f"请求失败: {last_error}")


# =============================
# Embedding & Chat API
# =============================

def embeddings(texts):
    res = _post_json(
        f"{BASE_URL}/embeddings",
        {"model": EMBEDDING_MODEL, "input": texts}
    )
    return [d["embedding"] for d in res["data"]]


def chat(messages, temperature=0.3):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    out = _post_json(f"{BASE_URL}/chat/completions", payload)
    content = out["choices"][0]["message"]["content"]
    # 确保返回的是 UTF-8 编码的字符串
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    elif not isinstance(content, str):
        content = str(content)
    return content


# =============================
# PDF 分段拆分
# =============================

def split_text(text, chunk=900, overlap=150):
    text = (text or "").replace("\r", "")
    out = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + chunk, n)
        part = text[i:j].strip()
        if part:
            out.append(part)
        if j == n:
            break
        i = max(0, j - overlap)
    return out


# =============================
# 构建索引
# =============================

def build_index():
    print("[ingest] loading PDF...")
    reader = PdfReader(DOC_PATH)
    chunks = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for ci, c in enumerate(split_text(text)):
            cid = hashlib.sha1(f"{i}:{ci}:{c[:20]}".encode()).hexdigest()
            chunks.append((cid, c, i+1))

    print(f"[ingest] total chunks = {len(chunks)}")

    # 先算一个 embedding 看维度
    dim = len(embeddings(["probe"])[0])
    index = faiss.IndexFlatIP(dim)

    metas = []
    vecs = []

    for i in tqdm(range(0, len(chunks), 64)):
        batch = chunks[i:i+64]
        ids = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        pages = [x[2] for x in batch]

        embs = embeddings(texts)
        vecs.extend(embs)
        metas.extend(
            [{"id": cid, "text": t, "page": p}
             for cid, t, p in zip(ids, texts, pages)]
        )

    vecs = np.array(vecs, dtype=np.float32)
    # L2 normalize
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    # ⭐ 关键修复：将向量添加到索引
    index.add(vecs)

    # 保存索引
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("✅ 索引已构建")


# 如果不存在索引，则自动构建
if not os.path.exists(f"{INDEX_DIR}/index.faiss"):
    build_index()


# =============================
# RAG 查询接口
# =============================

def rag_ask(query, top_k=10):
    # 加载索引
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    metas = [
        json.loads(l)
        for l in open(f"{INDEX_DIR}/meta.jsonl", encoding="utf-8")
    ]

    # 生成检索向量
    q = embeddings([query])[0]
    q = np.array([q], dtype=np.float32)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    # 检索
    D, I = index.search(q, top_k)
    hits = [metas[i] for i in I[0]]

    # 构造上下文
    ctx = "\n\n---\n\n".join(
        [f"[p.{h['page']}]\n{h['text']}" for h in hits]
    )

    # 调用 Chat 模型
    messages = [
        {"role": "system", "content": "你是 Kindle 智能客服助手。请根据提供的参考内容回答用户问题。如果参考内容中包含相关信息，请详细回答；如果参考内容不相关或信息不足，可以说明情况，但尽量基于参考内容提供有用的信息。"},
        {"role": "user", "content": f"问题：{query}\n\n参考内容（来自 Kindle 用户手册）：\n{ctx}\n\n请基于上述参考内容回答用户的问题。"}
    ]

    answer = chat(messages)
    return answer, hits
