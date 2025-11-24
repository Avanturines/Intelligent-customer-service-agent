#uvicorn api:app --reload --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag_backend import rag_ask

app = FastAPI()
# 允许跨域（否则你的网页 JS 会被阻止）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 你也可以改成你的网页域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ask")
def ask(q: str, top_k: int = 5):
    """
    聊天接口：
    输入：q（用户问题）
    输出：RAG回答 & 来源
    """
    answer, sources = rag_ask(q, top_k=top_k)
    # 使用 JSONResponse 确保 UTF-8 编码
    return JSONResponse(
        content={
            "answer": answer,
            "sources": sources
        },
        media_type="application/json; charset=utf-8"
    )

@app.get("/")
def home():
    return {"msg": "RAG API is running!"}
