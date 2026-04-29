from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.core.config import settings


@dataclass
class Chunk:
    text: str
    video: str
    start: float
    end: float


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.model = SentenceTransformer(settings.embed_model)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        dim = self.model.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        if isinstance(vectors, np.ndarray):
            return vectors.tolist()
        return [np.asarray(v).tolist() for v in vectors]

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        vectors = self.embed([c.text for c in chunk_list])
        points = [
            PointStruct(
                id=abs(hash((c.video, c.start, c.end, i))) % (2**63 - 1),
                vector=vec,
                payload={
                    "text": c.text,
                    "video": c.video,
                    "start": c.start,
                    "end": c.end,
                },
            )
            for i, (c, vec) in enumerate(zip(chunk_list, vectors, strict=True))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_vec = self.embed([query])[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=limit,
            with_payload=True,
        )
        results: list[dict] = []
        for hit in hits:
            if not hit.payload:
                continue
            payload = dict(hit.payload)
            payload["_score"] = float(hit.score or 0.0)
            results.append(payload)
        return results
