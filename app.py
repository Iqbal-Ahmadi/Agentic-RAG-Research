from rag.config import RAGConfig
from rag.vector_index import VectorIndex
from rag.groq_llm import GroqLLM
from rag.agent import AgenticRAG
import os

def main():
    cfg = RAGConfig()
    pdf_dir = "./papers"

    index = VectorIndex(cfg.embed_model_name)
    index.build_from_pdf_dir(pdf_dir, cfg.chunk_chars, cfg.chunk_overlap)

    llm = GroqLLM(cfg.groq_model)
    agent = AgenticRAG(cfg, index, llm)

    while True:
        q = input("\nAsk a research question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        try:
            ans = agent.answer(q)
            print("\n--- ANSWER ---\n")
            print(ans)
        except Exception as e:
            print(f"\nERROR: {e}")

if __name__ == "__main__":
    # print(os.environ.get('GROQ_API_KEY'))
    main()