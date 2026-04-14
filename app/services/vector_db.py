import asyncio
from typing import Optional
from actian_vectorai import (
    AsyncVectorAIClient,
    VectorParams,
    Distance,
    PointStruct,
    Field,
    FilterBuilder,
)
from app.config import get_settings

settings = get_settings()


class VectorDBService:
    def __init__(self, url: str = None):
        self.url = url or settings.vector_db_url
        self.collection_name = settings.collection_name
        self._client: Optional[AsyncVectorAIClient] = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = AsyncVectorAIClient(self.url)
            await self._client.connect()

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> dict:
        await self.connect()
        return await self._client.health_check()

    async def ensure_collection(self, dimension: int = None) -> bool:
        await self.connect()
        dimension = dimension or settings.embedding_dimension
        
        exists = await self._client.collections.exists(self.collection_name)
        if not exists:
            await self._client.collections.create(
                self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.Cosine
                )
            )
            return True
        return False

    async def collection_exists(self) -> bool:
        await self.connect()
        return await self._client.collections.exists(self.collection_name)

    async def get_collection_info(self) -> dict:
        await self.connect()
        if await self.collection_exists():
            return await self._client.collections.get_info(self.collection_name)
        return None

    async def upsert_point(
        self,
        vector: list[float],
        payload: dict,
        point_id: int = None
    ) -> int:
        await self.connect()
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        result = await self._client.points.upsert(self.collection_name, [point])
        return result

    async def upsert_batch(
        self,
        vectors: list[list[float]],
        payloads: list[dict],
        ids: list[int] = None
    ) -> int:
        await self.connect()
        points = [
            PointStruct(id=idx, vector=vec, payload=pay)
            for idx, (vec, pay) in enumerate(zip(vectors, payloads))
        ]
        if ids:
            for i, p in enumerate(points):
                p.id = ids[i]
        
        result = await self._client.points.upsert(self.collection_name, points)
        return result

    async def search(
        self,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = None
    ) -> list:
        await self.connect()
        results = await self._client.points.search(
            self.collection_name,
            vector=vector,
            limit=limit,
            score_threshold=score_threshold
        )
        return results

    async def get_by_id(self, point_id: int) -> dict:
        await self.connect()
        points = await self._client.points.get(self.collection_name, ids=[point_id])
        if points:
            return points[0].payload
        return None

    async def get_by_ids(self, point_ids: list[int]) -> list[dict]:
        await self.connect()
        points = await self._client.points.get(self.collection_name, ids=point_ids)
        return [(p.id, p.payload) for p in points]

    async def count(self) -> int:
        await self.connect()
        return await self._client.points.count(self.collection_name)

    async def delete_collection(self) -> None:
        await self.connect()
        if await self.collection_exists():
            await self._client.collections.delete(self.collection_name)

    async def delete_by_ids(self, point_ids: list[int]) -> None:
        await self.connect()
        await self._client.points.delete(self.collection_name, ids=point_ids)


vector_db = VectorDBService()
