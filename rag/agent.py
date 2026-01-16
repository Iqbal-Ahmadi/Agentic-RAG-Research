import re
from typing import Dict, List, Tuple

from .config import RAGConfig
from .safety import validate_user_question, validate_retrieval_params
from .vector_index import VectorIndex
from .groq_llm import GroqLLM

MAKER_SYSTEM = """
You are a Research Writer in an agentic RAG system.
Use ONLY the provided context excerpts.
Every key claim must include citations like [Paper.pdf p.X].
If the context is insufficient, say: "Insufficient evidence in the provided papers."
Do not invent references or facts.
Write in a helpful, concise research style.
"""

CHECKER_SYSTEM = """
You are a strict Research Critic.
You will be given: question, context excerpts, and a draft answer.

Check for:
1) Unsupported claims (not grounded in the context)
2) Missing/incorrect citations
3) Vague wording or incomplete coverage
4) Safety issues (PII, policy violations)

Return ONLY valid JSON:
{
  "verdict": "accept" or "revise",
  "critique": ["...","..."],
  "revision_instructions": "how to improve",
  "query_refinement": "optional improved retrieval query"
}
"""

class AgenticRAG:
    def __init__(self, config: RAGConfig, index: VectorIndex, llm: GroqLLM):
        self.cfg = config
        self.index = index
        self.llm = llm

    def answer(self, question: str) -> str:
        q = validate_user_question(question, self.cfg.max_question_chars)
        top_k = validate_retrieval_params(self.cfg.top_k)

        # First retrieval
        context, retrieved = self.index.retrieve(q, top_k=top_k)

        # Maker draft
        draft = self._maker(q, context)

        # Checker loop
        for _ in range(self.cfg.max_iterations):
            review = self._checker(q, context, draft)

            if review.get("verdict") == "accept":
                return self._output_guard(draft, retrieved)

            refined = (review.get("query_refinement") or "").strip()
            if refined:
                context, retrieved = self.index.retrieve(refined, top_k=top_k)

            draft = self._revise(review, context, draft)

        return self._output_guard(draft, retrieved)

    def _maker(self, q: str, context: str) -> str:
        user = f"Question:\n{q}\n\nContext excerpts:\n{context}\n\nWrite the answer with citations."
        return self.llm.chat(
            messages=[{"role": "system", "content": MAKER_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=self.cfg.maker_temperature,
            max_tokens=self.cfg.max_completion_tokens,
            stream=True,
        )

    def _checker(self, q: str, context: str, draft: str) -> Dict:
        user = f"Question:\n{q}\n\nContext excerpts:\n{context}\n\nDraft:\n{draft}\n\nReturn JSON."
        return self.llm.chat_json(
            messages=[{"role": "system", "content": CHECKER_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=self.cfg.checker_temperature,
            max_tokens=600,
        )

    def _revise(self, review: Dict, context: str, draft: str) -> str:
        revise_system = "Revise the draft using the instructions. Keep citations accurate and only from context."
        instructions = review.get("revision_instructions", "Improve clarity and grounding with citations.")
        critique = "\n".join(review.get("critique", []))

        revise_user = f"""Revision instructions:
{instructions}

Critique notes:
{critique}

Context excerpts:
{context}

Old draft:
{draft}

Return improved answer with correct citations only."""
        return self.llm.chat(
            messages=[{"role": "system", "content": revise_system},
                      {"role": "user", "content": revise_user}],
            temperature=self.cfg.maker_temperature,
            max_tokens=self.cfg.max_completion_tokens,
            stream=True,
        )

    def _output_guard(self, answer: str, retrieved: List[Dict]) -> str:
        """
        Output safety:
          - Prevent fabricated citations by checking referenced sources exist.
        """
        allowed_sources = {d["source"] for d in retrieved}
        cited = re.findall(r"\[([^\]]+)\]", answer)
        cited_sources = set()

        for c in cited:
            # expects "Paper.pdf p.X"
            if " p." in c:
                cited_sources.add(c.split(" p.")[0].strip())

        bad = cited_sources - allowed_sources
        if bad:
            return (
                "I found relevant text, but I couldn't verify some citations against the retrieved papers. "
                "Please retry or add more PDFs to the corpus."
            )

        # If no citations at all, force safer response
        if not cited_sources and len(retrieved) > 0:
            return (
                "Insufficient evidence in the provided papers to answer confidently with citations. "
                "Try adding more papers or refining the question."
            )

        return answer
