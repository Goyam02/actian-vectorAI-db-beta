import uuid
from typing import Optional
from app.config import get_settings
from app.services.vector_db import vector_db
from app.services.embeddings import embedding_service

settings = get_settings()

_next_id = 0


def _get_next_id(count: int) -> int:
    global _next_id
    if _next_id == 0:
        _next_id = count
    return _next_id


def _increment_id(n: int) -> None:
    global _next_id
    _next_id += n


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[dict]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    full_text = "\n\n".join(paragraphs)
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(full_text):
        end = start + chunk_size
        
        if end < len(full_text):
            space_idx = full_text.rfind(' ', start, end)
            if space_idx > start:
                end = space_idx
        else:
            end = len(full_text)
        
        chunk_text_str = full_text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "char_count": len(chunk_text_str),
            })
            chunk_id += 1
        
        start = end - overlap if end < len(full_text) else len(full_text)
    
    return chunks


async def index_document(text: str, metadata: dict = None, doc_id: str = None) -> tuple[int, list[dict]]:
    chunks = chunk_text(text)
    texts = [chunk["text"] for chunk in chunks]
    
    embeddings = embedding_service.encode(texts)
    
    current_count = await vector_db.count()
    doc_id = doc_id or str(uuid.uuid4())[:8]
    start_id = _get_next_id(current_count)
    
    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "text": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "char_count": chunk["char_count"],
            "document_id": doc_id,
        }
        if metadata:
            payload["metadata"] = metadata
        payloads.append(payload)
    
    ids = [start_id + i for i in range(len(chunks))]
    
    await vector_db.upsert_batch(embeddings, payloads, ids)
    _increment_id(len(chunks))
    
    return len(chunks), chunks


async def search_documents(
    query: str,
    top_k: int = None,
    score_threshold: float = None
) -> list[dict]:
    top_k = top_k or settings.default_top_k
    
    query_embedding = embedding_service.encode_single(query)
    
    results = await vector_db.search(
        vector=query_embedding,
        limit=top_k,
        score_threshold=score_threshold
    )
    
    search_results = []
    for r in results:
        payload = r.payload
        search_results.append({
            "id": r.id,
            "score": r.score,
            "text": payload.get("text", ""),
            "metadata": payload.get("metadata", {}),
            "document_id": payload.get("document_id", "")
        })
    
    return search_results


def build_context(results: list[dict]) -> str:
    context_parts = []
    for i, result in enumerate(results, 1):
        doc_label = f"[Doc: {result.get('document_id', 'unknown')}] "
        context_parts.append(f"{doc_label}{result['text']}")
    return "\n\n".join(context_parts)


async def generate_answer_local(query: str, context: str) -> str:
    return f"""Based on the retrieved context, here's what I found:

Query: {query}

Relevant Information:
{context}

Note: This is using simple context retrieval. For full answer generation, 
integrate with an LLM (OpenAI, Anthropic, Ollama, etc.)."""


async def generate_answer_azure_openai(query: str, context: str) -> str:
    if not settings.azure_openai_endpoint:
        return await generate_answer_local(query, context)
    
    try:
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"}
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        
        return response.choices[0].message.content
    except ImportError:
        return await generate_answer_local(query, context)
    except Exception as e:
        return f"Error calling Azure OpenAI API: {str(e)}\n\nContext:\n{context}"


async def rag_query(
    query: str,
    top_k: int = None,
    use_llm: bool = None
) -> tuple[str, list[dict], str]:
    use_llm = use_llm if use_llm is not None else True
    
    results = await search_documents(query, top_k)
    
    context = build_context(results)
    
    if use_llm:
        if settings.azure_openai_endpoint:
            answer = await generate_answer_azure_openai(query, context)
        elif settings.openai_api_key:
            answer = await generate_answer_openai(query, context)
        else:
            answer = await generate_answer_local(query, context)
    else:
        answer = await generate_answer_local(query, context)
    
    return answer, results, context


async def generate_answer_openai(query: str, context: str, api_key: str = None) -> str:
    api_key = api_key or settings.openai_api_key
    if not api_key:
        return await generate_answer_local(query, context)
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"}
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        
        return response.choices[0].message.content
    except ImportError:
        return await generate_answer_local(query, context)
    except Exception as e:
        return f"Error calling LLM API: {str(e)}\n\nContext:\n{context}"


async def get_all_documents() -> dict:
    count = await vector_db.count()
    return {"total_chunks": count}


async def reset_document_counter() -> None:
    global _next_id
    _next_id = 0
