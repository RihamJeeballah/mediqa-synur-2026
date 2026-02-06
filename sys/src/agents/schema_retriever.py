# src/agents/schema_retriever.py
import numpy as np
from typing import Dict, List, Any
import ollama


class SchemaRetriever:
    def __init__(
        self,
        schema_by_id: Dict[str, Dict[str, Any]],
        embed_model: str = "nomic-embed-text",
        top_k: int = 40,
    ):
        self.schema_by_id = schema_by_id
        self.embed_model = embed_model
        self.top_k = top_k

        # Build schema texts
        self.schema_ids: List[str] = []
        self.schema_texts: List[str] = []

        for cid, s in schema_by_id.items():
            text = f"{s.get('name','')} {s.get('value_type','')} {' '.join(s.get('value_enum', []))}"
            self.schema_ids.append(cid)
            self.schema_texts.append(text)

        # Precompute embeddings once
        self.schema_embeddings = self._embed_texts(self.schema_texts)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for t in texts:
            res = ollama.embeddings(
                model=self.embed_model,
                prompt=t,
            )
            embeddings.append(res["embedding"])
        return np.array(embeddings)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b, a)

    def retrieve(self, transcript_chunk: str) -> List[str]:
        res = ollama.embeddings(
            model=self.embed_model,
            prompt=transcript_chunk,
        )
        chunk_emb = np.array(res["embedding"])

        sims = self._cosine_sim(chunk_emb, self.schema_embeddings)
        top_idx = np.argsort(sims)[::-1][: self.top_k]

        return [self.schema_ids[i] for i in top_idx]
