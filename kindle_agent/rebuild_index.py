"""
重新构建 RAG 索引
运行此脚本将重新构建向量索引（需要几分钟时间）
"""
from rag_backend import build_index

if __name__ == '__main__':
    print("=" * 50)
    print("开始重新构建索引...")
    print("=" * 50)
    print()
    
    build_index()
    
    print()
    print("=" * 50)
    print("✅ 索引构建完成！现在可以启动应用了")
    print("=" * 50)

