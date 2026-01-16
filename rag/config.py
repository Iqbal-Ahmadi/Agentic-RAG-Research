from dataclasses import dataclass

@dataclass(frozen=True)
class RAGConfig:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    chunk_chars: int = 1100
    chunk_overlap: int = 200
    max_question_chars: int = 2000
    max_iterations: int = 2

    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    maker_temperature: float = 0.2
    checker_temperature: float = 0.0
    max_completion_tokens: int = 900
