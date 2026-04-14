from fastapi import APIRouter, HTTPException
from app.models import (
    DocumentUpload,
    SearchRequest,
    SearchResult,
    RAGRequest,
    RAGResponse,
    DocumentIndexResponse,
)
from app.services.vector_db import vector_db
from app.services.rag import (
    index_document,
    search_documents,
    rag_query,
    get_all_documents,
    reset_document_counter
)

router = APIRouter()


@router.post("/documents", response_model=DocumentIndexResponse)
async def upload_document(doc: DocumentUpload):
    try:
        chunks_count, _ = await index_document(doc.text, doc.metadata, str(doc.id) if doc.id else None)
        total = await vector_db.count()
        return DocumentIndexResponse(
            indexed_count=chunks_count,
            collection=vector_db.collection_name,
            total_chunks=total
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")


@router.get("/documents/stats")
async def get_stats():
    try:
        info = await get_all_documents()
        collection_info = await vector_db.get_collection_info()
        return {
            "total_chunks": info["total_chunks"],
            "collection": vector_db.collection_name,
            "collection_info": str(collection_info) if collection_info else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    try:
        results = await search_documents(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        return [SearchResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    try:
        answer, sources, context = await rag_query(
            query=request.query,
            top_k=request.top_k,
            use_llm=request.use_llm
        )
        return RAGResponse(
            query=request.query,
            answer=answer,
            sources=[SearchResult(**s) for s in sources],
            context_used=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    try:
        await vector_db.delete_by_ids([doc_id])
        return {"status": "deleted", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.delete("/collection")
async def delete_collection():
    try:
        await vector_db.delete_collection()
        await reset_document_counter()
        return {"status": "deleted", "collection": vector_db.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete collection failed: {str(e)}")


@router.post("/collection/reset")
async def reset_collection():
    try:
        await vector_db.delete_collection()
        await vector_db.ensure_collection()
        await reset_document_counter()
        return {"status": "reset", "collection": vector_db.collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
