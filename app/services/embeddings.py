from typing import Optional
from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def _load_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_embedding_dimension()

    def encode(self, texts: str | list[str]) -> list[list[float]]:
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def encode_single(self, text: str) -> list[float]:
        return self.encode(text)[0]


embedding_service = EmbeddingService()
