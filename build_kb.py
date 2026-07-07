"""
预构建向量知识库 — 运行一次，Streamlit 秒开
用法: python build_kb.py
"""
from rag_pipeline import RAGAssistant

print("=" * 60)
print("🔨 预构建向量知识库")
print("=" * 60)

assistant = RAGAssistant()
assistant.build_knowledge_base()

print("\n✅ 向量库已保存在 data/chroma_db/")
print("   现在启动 Streamlit 即可秒开!")
