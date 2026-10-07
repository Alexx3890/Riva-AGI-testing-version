"""Qdrant Vector Database Storage Backend for rag_knowledge.

Provides semantic vector search, payload filtering, and fast document retrieval
via Qdrant Cloud and local FastEmbed (BAAI/bge-small-en-v1.5) ONNX embeddings.
"""

import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag.storage.qdrant")

_NAMESPACE_RIVA = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_ID_TOKEN_PATTERN = re.compile(r"^(?=.*\d)[A-Za-z0-9]{6,}$")  # Alphanumeric with at least 1 digit, min 6 chars

try:
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False


def string_to_point_id(doc_id: str) -> str:
    """Generates a deterministic UUID string from an arbitrary document string ID."""
    clean_id = str(doc_id).strip()
    return str(uuid.uuid5(_NAMESPACE_RIVA, clean_id))


class QdrantKnowledgeStore:
    """Manages Qdrant Cloud connection, vector embeddings, and semantic knowledge retrieval."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 4.0,
    ) -> None:
        try:
            from rag_knowledge import load_env
            load_env()
        except ImportError:
            pass

        self.url = (
            url
            if url is not None
            else (
                os.getenv("QDRANT_URL", "").strip()
                or os.getenv("QDRANT_CLUSTER_ENDPOINT", "").strip()
                or os.getenv("qdrant_cluster_endpoint", "").strip()
            )
        )
        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.getenv("QDRANT_WRITE_API_KEY", "").strip()
                or os.getenv("QDRANT_API_KEY", "").strip()
                or os.getenv("qdrant_api", "").strip()
            )
        )
        self.collection_name = (
            collection_name
            if collection_name is not None
            else (
                os.getenv("QDRANT_COLLECTION", "riva_knowledge").strip()
                or "riva_knowledge"
            )
        )
        self.model_name = (
            model_name
            if model_name is not None
            else (os.getenv("EMBEDDING_MODEL", "").strip() or "BAAI/bge-small-en-v1.5")
        )
        self.vector_size = 384  # Dimension for BAAI/bge-small-en-v1.5
        self.timeout = timeout

        self._client: Optional[Any] = None
        self._embedding_model: Optional[Any] = None
        self._is_connected: Optional[bool] = None
        self._retry_after: float = 0.0
        self._cooldown_seconds: float = 30.0
        self._lock = threading.Lock()

    def _get_embedding_model(self) -> Any:
        """Lazy-loads the FastEmbed ONNX model."""
        if self._embedding_model is None:
            with self._lock:
                if self._embedding_model is None:
                    logger.info(f"Loading FastEmbed model '{self.model_name}'...")
                    self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def connect(self) -> bool:
        """Initializes client and verifies connectivity to Qdrant cluster."""
        with self._lock:
            if not _HAS_QDRANT:
                logger.warning("qdrant-client or fastembed not installed.")
                self._is_connected = False
                self._retry_after = time.monotonic() + self._cooldown_seconds
                return False

            if not self.url or not self.api_key:
                logger.warning("Qdrant URL or API key missing in environment.")
                self._is_connected = False
                self._retry_after = time.monotonic() + self._cooldown_seconds
                return False

            try:
                self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
                # Verify connection
                self._client.get_collections()
                self._ensure_collection()
                self._is_connected = True
                self._retry_after = 0.0
                logger.info(f"Connected to Qdrant Cloud successfully (collection: '{self.collection_name}')")
                return True
            except ValueError:
                self._is_connected = False
                raise
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant Cloud at {self.url}: {e}", exc_info=True)
                self._is_connected = False
                self._retry_after = time.monotonic() + self._cooldown_seconds
                return False

    def _ensure_collection(self) -> None:
        """Ensures the vector collection and payload indexes exist with dimension validation."""
        if not self._client:
            return

        collections = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (vector size: {self.vector_size})...")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            # Dimension check on existing collection
            try:
                col_info = self._client.get_collection(self.collection_name)
                vectors_cfg = getattr(col_info.config.params, "vectors", None)
                existing_size = getattr(vectors_cfg, "size", None)
                if existing_size is None and isinstance(vectors_cfg, dict):
                    existing_size = getattr(next(iter(vectors_cfg.values())), "size", None)
                if existing_size and existing_size != self.vector_size:
                    raise ValueError(
                        f"Collection '{self.collection_name}' vector dimension is {existing_size}, "
                        f"but model '{self.model_name}' requires {self.vector_size} dimensions."
                    )
            except Exception as ex:
                if isinstance(ex, ValueError):
                    raise
                logger.debug("Dimension check bypassed or collection inspection note: %s", ex)

        # Idempotently ensure payload indexes exist (runs on new and existing collections)
        for field, field_type in [
            ("is_active", models.PayloadSchemaType.BOOL),
            ("aliases", models.PayloadSchemaType.KEYWORD),
            ("id", models.PayloadSchemaType.KEYWORD),
            ("category", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=field_type,
                )
            except Exception as ex:
                logger.debug(f"Payload index creation note on '{field}': {ex}")

    def is_available(self) -> bool:
        """Checks if Qdrant is available with cooldown retry protection."""
        if self._is_connected is True:
            return True
        if self._is_connected is False and time.monotonic() < self._retry_after:
            return False
        ok = self.connect()
        if not ok:
            self._retry_after = time.monotonic() + self._cooldown_seconds
        return ok

    def _build_embed_text(self, doc: Dict[str, Any]) -> str:
        """Constructs rich text representation combining title, summary, aliases, keywords, and content."""
        text_parts = [
            doc.get("title", ""),
            doc.get("summary", ""),
            " ".join(doc.get("aliases", [])),
            " ".join(doc.get("keywords", [])),
            doc.get("content", ""),
        ]
        return "\n".join(p for p in text_parts if p).strip()

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Embeds and upserts a list of valid documents into Qdrant collection."""
        if not self.is_available() or not self._client:
            raise RuntimeError("Qdrant store is not available for upsert operation.")

        valid_docs = [d for d in documents if d.get("id") or d.get("_id")]
        if not valid_docs:
            return 0

        model = self._get_embedding_model()
        embed_texts = [self._build_embed_text(d) for d in valid_docs]
        vectors = list(model.embed(embed_texts))

        points = []
        for doc, vector in zip(valid_docs, vectors):
            doc_id = str(doc.get("id") or doc.get("_id"))
            point_id = string_to_point_id(doc_id)

            payload = {
                "id": doc_id,
                "title": doc.get("title", ""),
                "aliases": doc.get("aliases", []),
                "keywords": doc.get("keywords", []),
                "summary": doc.get("summary", ""),
                "content": doc.get("content", ""),
                "category": doc.get("category", "general"),
                "metadata": doc.get("metadata", {}),
                "is_active": bool(doc.get("is_active", True)),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} points into Qdrant collection '{self.collection_name}'")
        return len(points)

    def search_text(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Performs hybrid semantic vector search + exact identifier matching with is_active filtering."""
        clean_query = query.strip()
        if not clean_query or not self.is_available() or not self._client:
            return []

        active_filter = models.FieldCondition(key="is_active", match=models.MatchValue(value=True))

        try:
            model = self._get_embedding_model()
            query_vector = list(model.embed([clean_query]))[0]
            q_vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

            # 1. Semantic Vector Query (active documents only)
            search_results = self._client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                query_filter=models.Filter(must=[active_filter]),
                limit=top_k * 2,
                score_threshold=min_score,
                with_payload=True,
            ).points

            # 2. Check for exact identifier matches (Roll Number, UID) in query tokens
            candidate_tokens = [re.sub(r"[^A-Za-z0-9]", "", t).upper() for t in clean_query.split()]
            id_tokens = list({t for t in candidate_tokens if _ID_TOKEN_PATTERN.match(t)})

            exact_matches = []
            if id_tokens:
                try:
                    exact_filter = models.Filter(
                        must=[
                            active_filter,
                            models.Filter(
                                should=[
                                    models.FieldCondition(key="aliases", match=models.MatchAny(any=id_tokens)),
                                    models.FieldCondition(
                                        key="id",
                                        match=models.MatchAny(any=[f"student_{i.lower()}" for i in id_tokens]),
                                    ),
                                ]
                            ),
                        ]
                    )
                    hits = self._client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=exact_filter,
                        limit=top_k,
                        with_payload=True,
                    )[0]
                    exact_matches.extend(hits)
                except Exception as ex:
                    logger.debug(f"Exact match check exception: {ex}")

            # Combine exact matches (boosted to 100.0) and semantic matches
            combined: Dict[str, Dict[str, Any]] = {}

            for hit in exact_matches:
                payload = hit.payload or {}
                doc_id = payload.get("id") or str(hit.id)
                meta = dict(payload.get("metadata") or {})
                meta.pop("private", None)
                combined[doc_id] = {
                    "id": doc_id,
                    "title": payload.get("title", ""),
                    "summary": payload.get("summary", ""),
                    "content": payload.get("content", ""),
                    "metadata": meta,
                    "score": 100.0,
                }

            for hit in search_results:
                payload = hit.payload or {}
                doc_id = payload.get("id") or str(hit.id)
                if doc_id not in combined:
                    meta = dict(payload.get("metadata") or {})
                    meta.pop("private", None)
                    combined[doc_id] = {
                        "id": doc_id,
                        "title": payload.get("title", ""),
                        "summary": payload.get("summary", ""),
                        "content": payload.get("content", ""),
                        "metadata": meta,
                        "score": round(float(hit.score) * 100, 2),
                    }

            results = list(combined.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error executing Qdrant search for '{clean_query}': {e}", exc_info=True)
            self._is_connected = False
            self._retry_after = time.monotonic() + self._cooldown_seconds
            return []

    def list_documents(self, limit: int = 100, offset: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Lists active documents from Qdrant with optional pagination offset."""
        if not self.is_available() or not self._client:
            return []

        active_filter = models.FieldCondition(key="is_active", match=models.MatchValue(value=True))

        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=[active_filter]),
                limit=limit,
                offset=offset,
                with_payload=True,
            )
            docs = []
            for p in points:
                payload = p.payload or {}
                meta = dict(payload.get("metadata") or {})
                meta.pop("private", None)
                docs.append({
                    "id": payload.get("id", str(p.id)),
                    "title": payload.get("title", ""),
                    "summary": payload.get("summary", ""),
                    "keywords": payload.get("keywords", []),
                    "aliases": payload.get("aliases", []),
                    "category": payload.get("category", "general"),
                    "metadata": meta,
                    "is_active": payload.get("is_active", True),
                })
            return docs
        except Exception as e:
            logger.error(f"Error listing documents from Qdrant: {e}", exc_info=True)
            return []

    def get_document(self, doc_id: str, include_private: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieves a single document by doc_id.

        Args:
            doc_id: The document identifier (e.g. 'student_250001010001').
            include_private: If True, returns private metadata fields for admin/script use.
                             Defaults to False to prevent exposing sensitive data.
        """
        if not self.is_available() or not self._client:
            return None

        point_id = string_to_point_id(doc_id)
        try:
            points = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            )
            if not points:
                return None
            payload = points[0].payload or {}
            meta = dict(payload.get("metadata") or {})
            if not include_private:
                meta.pop("private", None)

            return {
                "id": payload.get("id", str(points[0].id)),
                "title": payload.get("title", ""),
                "summary": payload.get("summary", ""),
                "content": payload.get("content", ""),
                "keywords": payload.get("keywords", []),
                "aliases": payload.get("aliases", []),
                "category": payload.get("category", "general"),
                "metadata": meta,
                "is_active": payload.get("is_active", True),
            }
        except Exception as e:
            logger.error(f"Error retrieving document '{doc_id}' from Qdrant: {e}", exc_info=True)
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document by ID."""
        if not self.is_available() or not self._client:
            return False

        point_id = string_to_point_id(doc_id)
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id],
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}' from Qdrant: {e}", exc_info=True)
            return False


_global_qdrant_store: Optional[QdrantKnowledgeStore] = None
_global_qdrant_lock = threading.Lock()


def get_global_qdrant_store() -> QdrantKnowledgeStore:
    """Returns the process-wide QdrantKnowledgeStore singleton."""
    global _global_qdrant_store
    if _global_qdrant_store is None:
        with _global_qdrant_lock:
            if _global_qdrant_store is None:
                _global_qdrant_store = QdrantKnowledgeStore()
    return _global_qdrant_store
