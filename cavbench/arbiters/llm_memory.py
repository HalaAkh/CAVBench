"""Memory module of the faithful CoDrivingLLM mode (retrieval-augmented generation).

CoDrivingLLM keeps a memory of past decisions: after each step it judges
whether a CAV's action made a conflict worse, and stores

    scenario description -> action -> feedback

Before each decision call, the most similar past scenarios (cosine similarity
of text embeddings) are added to the prompt as examples ("few-shot").

This module is written from the paper's description (no code is copied).
Differences from the original: our own JSONL file instead of a vector
database, and a free local embedding for the "mock" model and the tests.

Embeddings:
* ``"local"``: a hashed bag-of-words vector computed with numpy. Free, needs no
  key, deterministic across processes (crc32, not Python's salted ``hash``).
* ``"openai"``: OpenAI ``text-embedding-3-small`` (same ``OPENAI_API_KEY``);
  its calls and cost are counted.

A memory file remembers which embedding built it; loading it with another
embedding raises an error, because the vectors would not be comparable.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np

MEMORY_DIR = Path(__file__).resolve().parents[2] / "results" / "llm_memory"
EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE = 0.02  # USD per million tokens (cost report only)
LOCAL_DIM = 256

FEEDBACK_BAD = "Your action intensified the conflict; similar actions should be avoided."
FEEDBACK_CRASH = "Your action was followed by a collision; similar actions should be avoided."
FEEDBACK_GOOD = "This action resolved the conflict safely."


def local_embedding(text: str, dim: int = LOCAL_DIM) -> np.ndarray:
    """Hashed bag of words (and word pairs), L2-normalised."""
    words = re.findall(r"[a-z]+|\d+(?:\.\d)?", text.lower())
    vec = np.zeros(dim)
    for tok in words + [a + "_" + b for a, b in zip(words, words[1:])]:
        vec[zlib.crc32(tok.encode("utf-8")) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class LLMMemory:
    """A list of past experiences with their embedding vectors."""

    def __init__(self, embedding: str = "local", k: int = 2) -> None:
        if embedding not in ("local", "openai"):
            raise ValueError(f"embedding must be 'local' or 'openai', got {embedding!r}")
        self.embedding, self.k = embedding, k
        self.items: list[dict] = []
        self._vectors: list[np.ndarray] = []
        self._embed_cache: dict[str, np.ndarray] = {}
        self._client = None
        # usage counters (cumulative; the arbiter reports per-episode differences)
        self.embed_calls = self.embed_tokens = 0
        self.hits = 0

    # -- embeddings ---------------------------------------------------------
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Vectors for ``texts``; with OpenAI, all new texts go in ONE request."""
        new = [t for t in dict.fromkeys(texts) if t not in self._embed_cache]
        if new:
            if self.embedding == "local":
                for t in new:
                    self._embed_cache[t] = local_embedding(t)
            else:
                for t, v in zip(new, self._openai_embed(new)):
                    self._embed_cache[t] = v
        return [self._embed_cache[t] for t in texts]

    def _openai_embed(self, texts: list[str]) -> list[np.ndarray]:
        if self._client is None:
            from .codrivingllm_wrapper import require_api_key

            require_api_key()
            from openai import OpenAI  # optional dependency: pip install openai

            self._client = OpenAI(timeout=30.0, max_retries=5)
        r = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        self.embed_calls += 1
        if getattr(r, "usage", None) is not None:
            self.embed_tokens += r.usage.total_tokens
        out = []
        for d in r.data:
            v = np.asarray(d.embedding, dtype=float)
            out.append(v / max(np.linalg.norm(v), 1e-12))
        return out

    @property
    def embed_cost_usd(self) -> float:
        return self.embed_tokens * EMBED_PRICE / 1e6

    # -- store / retrieve ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def add(self, text: str, action: str, feedback: str, vector: np.ndarray | None = None) -> None:
        if vector is None:
            vector = self.embed([text])[0]
        self.items.append({"text": text, "action": action, "feedback": feedback})
        self._vectors.append(np.asarray(vector, dtype=float))

    def retrieve(self, text: str, k: int | None = None, vector: np.ndarray | None = None) -> list[dict]:
        """The ``k`` most similar stored items (cosine similarity), best first."""
        k = self.k if k is None else k
        if not self.items or k <= 0:
            return []
        q = self.embed([text])[0] if vector is None else vector
        sims = np.stack(self._vectors) @ q  # vectors are normalised
        best = np.argsort(-sims, kind="stable")[:k]
        self.hits += len(best)
        return [dict(self.items[i], similarity=float(sims[i])) for i in best]

    # -- files --------------------------------------------------------------
    @staticmethod
    def path_for(name: str) -> Path:
        return MEMORY_DIR / f"{name}.jsonl"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item, vec in zip(self.items, self._vectors):
                row = dict(item, embedding=self.embedding,
                           vector=[round(float(x), 6) for x in vec])
                f.write(json.dumps(row) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, embedding: str | None = None, k: int = 2) -> "LLMMemory":
        path = Path(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        found = rows[0]["embedding"] if rows else (embedding or "local")
        if embedding is not None and found != embedding:
            raise ValueError(f"{path} was built with {found!r} embeddings, not {embedding!r}; "
                             f"rebuild it with experiments/build_llm_memory.py")
        mem = cls(found, k)
        for r in rows:
            mem.add(r["text"], r["action"], r["feedback"], np.asarray(r["vector"], dtype=float))
        return mem

