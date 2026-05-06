"""Memory store using Qdrant — project-isolated, with time-decay and limits.

Architecture:
- Each project gets its own collection: "mem_{project}"
- "mem_general" for non-project-specific memories
- "mem_entities" for extracted entities (shared across projects, tagged)
- Time-decay: newer memories score higher
- Auto-cleanup: cap per collection (default 5000)
"""
import uuid
import time
import math
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    models
)

from config import (
    QDRANT_HOST, QDRANT_PORT, EMBED_DIM, MIN_SCORE,
    MAX_MEMORIES_INJECT, MAX_MEMORIES_PER_COLLECTION
)
from encoder import Encoder
from chunker import Chunk

ENTITY_COLLECTION = "mem_entities"


class MemoryStore:
    """Qdrant-backed memory store with project isolation."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        # Ensure entity collection exists
        self._ensure_collection(ENTITY_COLLECTION)
        # Ensure general collection
        self._ensure_collection("mem_general")
        print(f"[Memory] Connected to Qdrant")

    def _collection_name(self, project: str) -> str:
        """Get collection name for a project."""
        safe = project.lower().replace(" ", "_").replace("-", "_")
        return f"mem_{safe}"

    def _ensure_collection(self, name: str):
        """Create collection if it doesn't exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if name not in collections:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
                )
                # Create indexes
                for field in ["conv_id", "speaker", "project"]:
                    try:
                        self.client.create_payload_index(
                            collection_name=name,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD
                        )
                    except Exception:
                        pass
                print(f"[Memory] Created collection '{name}'")
        except Exception as e:
            print(f"[Memory] Error ensuring collection {name}: {e}")

    def store(self, chunks: list[Chunk], project: str = "general") -> int:
        """Store chunks in the project's collection. Returns count stored."""
        if not chunks:
            return 0

        col = self._collection_name(project)
        self._ensure_collection(col)

        points = []
        for chunk in chunks:
            embedding = self.encoder.encode(chunk.text)
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk.text,
                "speaker": chunk.speaker,
                "chunk_idx": chunk.chunk_idx,
                "conv_id": chunk.metadata.get("conv_id", ""),
                "project": project,
                "timestamp": time.time(),
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        self.client.upsert(collection_name=col, points=points)

        # Check limits
        self._check_limits(col)

        return len(points)

    def store_entities(self, entities: list, project: str = "general") -> int:
        """Store extracted entities in the shared entity collection."""
        if not entities:
            return 0

        self._ensure_collection(ENTITY_COLLECTION)

        points = []
        for ent in entities:
            text = ent.to_chunk_text()
            embedding = self.encoder.encode(text)
            point_id = str(uuid.uuid4())
            payload = {
                "text": text,
                "entity_type": ent.entity_type,
                "entity_value": ent.value,
                "context": ent.context,
                "importance": ent.importance,
                "project": project,
                "speaker": "entity",
                "timestamp": time.time(),
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        if points:
            self.client.upsert(collection_name=ENTITY_COLLECTION, points=points)

        return len(points)

    def recall(self, query: str, project: str = "general",
               top_k: int = MAX_MEMORIES_INJECT,
               exclude_conv_id: Optional[str] = None) -> list[dict]:
        """Recall from project collection + entities, with time-decay.
        
        When project is "general", searches ALL collections to find anything relevant.
        When project is specific, searches that project + general + entities.
        """
        query_emb = self.encoder.encode(query)
        now = time.time()

        all_results = []

        if project == "general":
            # No specific project → search ALL collections
            try:
                collections = [c.name for c in self.client.get_collections().collections]
                for col in collections:
                    if col.startswith("mem_") and col != ENTITY_COLLECTION:
                        hits = self._search(col, query_emb, top_k * 2, exclude_conv_id)
                        all_results.extend(hits)
            except Exception:
                pass
        else:
            # Specific project → search project + general
            col = self._collection_name(project)
            self._ensure_collection(col)
            hits = self._search(col, query_emb, top_k * 3, exclude_conv_id)
            all_results.extend(hits)

            gen_hits = self._search("mem_general", query_emb, top_k, exclude_conv_id)
            all_results.extend(gen_hits)

        # Always search entities
        entity_hits = self._search_entities(query_emb, project, top_k)
        all_results.extend(entity_hits)

        # Apply time-decay and deduplicate
        scored = []
        seen_texts = set()
        for hit in all_results:
            text = hit["text"]
            if text in seen_texts:
                continue
            seen_texts.add(text)

            # Time-decay: newer memories get a boost
            age_hours = (now - hit.get("timestamp", now)) / 3600
            decay = self._time_decay(age_hours)
            final_score = hit["score"] * decay

            if final_score >= MIN_SCORE:
                hit["score"] = round(final_score, 4)
                scored.append(hit)

        # Sort by final score
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _search(self, collection: str, query_emb: list, limit: int,
                exclude_conv_id: Optional[str] = None) -> list[dict]:
        """Search a single collection."""
        search_filter = None
        if exclude_conv_id:
            search_filter = Filter(
                must_not=[
                    FieldCondition(key="conv_id", match=MatchValue(value=exclude_conv_id))
                ]
            )

        try:
            from qdrant_client.models import QueryRequest
            results = self.client.query_points(
                collection_name=collection,
                query=query_emb,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                score_threshold=0.10  # Low threshold, time-decay handles filtering
            ).points
        except (ImportError, AttributeError, TypeError):
            results = self.client.search(
                collection_name=collection,
                query_vector=query_emb,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                score_threshold=0.10
            )

        hits = []
        for r in results:
            hits.append({
                "text": r.payload.get("text", ""),
                "score": r.score,
                "speaker": r.payload.get("speaker", ""),
                "timestamp": r.payload.get("timestamp", 0),
                "project": r.payload.get("project", "general"),
                "entity_type": r.payload.get("entity_type", ""),
            })
        return hits

    def _search_entities(self, query_emb: list, project: str, limit: int) -> list[dict]:
        """Search entities. When general, search all; otherwise filter by project."""
        search_filter = None
        if project != "general":
            search_filter = Filter(
                should=[
                    FieldCondition(key="project", match=MatchValue(value=project)),
                    FieldCondition(key="project", match=MatchValue(value="general")),
                ]
            )

        try:
            from qdrant_client.models import QueryRequest
            results = self.client.query_points(
                collection_name=ENTITY_COLLECTION,
                query=query_emb,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                score_threshold=0.10
            ).points
        except (ImportError, AttributeError, TypeError):
            results = self.client.search(
                collection_name=ENTITY_COLLECTION,
                query_vector=query_emb,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                score_threshold=0.10
            )

        hits = []
        for r in results:
            hits.append({
                "text": r.payload.get("text", ""),
                "score": r.score * r.payload.get("importance", 0.8),
                "speaker": "entity",
                "timestamp": r.payload.get("timestamp", 0),
                "project": r.payload.get("project", "general"),
                "entity_type": r.payload.get("entity_type", ""),
            })
        return hits

    def _time_decay(self, age_hours: float) -> float:
        """Time-decay factor: 1.0 for recent, decays smoothly."""
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.97
        elif age_hours < 168:  # 1 week
            return 0.90
        elif age_hours < 720:  # 1 month
            return 0.80
        else:
            return 0.70

    def _check_limits(self, collection: str):
        """Cleanup if collection exceeds max size."""
        try:
            info = self.client.get_collection(collection)
            count = info.points_count
            if count > MAX_MEMORIES_PER_COLLECTION:
                excess = count - MAX_MEMORIES_PER_COLLECTION + 100  # Delete 100 extra
                # Delete oldest points
                result = self.client.scroll(
                    collection_name=collection,
                    limit=excess,
                    order_by=models.OrderBy(key="timestamp", direction=models.Direction.ASC),
                    with_payload=False,
                )
                ids_to_delete = [p.id for p in result[0]]
                if ids_to_delete:
                    self.client.delete(
                        collection_name=collection,
                        points_selector=models.PointIdsList(points=ids_to_delete)
                    )
                    print(f"[Memory] Cleaned {len(ids_to_delete)} old memories from {collection}")
        except Exception as e:
            # Non-fatal — just log
            print(f"[Memory] Limit check error for {collection}: {e}")

    def count(self, project: str = None) -> int:
        """Get total memories. If project given, count for that project only."""
        try:
            total = 0
            collections = [c.name for c in self.client.get_collections().collections]
            for col in collections:
                if col.startswith("mem_"):
                    if project and col != self._collection_name(project):
                        continue
                    info = self.client.get_collection(col)
                    total += info.points_count
            return total
        except Exception:
            return 0

    def stats(self) -> dict:
        """Get per-project memory statistics."""
        result = {}
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            for col in collections:
                if col.startswith("mem_"):
                    info = self.client.get_collection(col)
                    project = col.replace("mem_", "")
                    result[project] = info.points_count
        except Exception as e:
            result["error"] = str(e)
        return result

    def clear_project(self, project: str) -> int:
        """Clear all memories for a specific project."""
        col = self._collection_name(project)
        try:
            info = self.client.get_collection(col)
            count = info.points_count
            self.client.delete_collection(col)
            self._ensure_collection(col)
            return count
        except Exception:
            return 0

    def clear_all(self):
        """Clear ALL memory collections."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            for col in collections:
                if col.startswith("mem_"):
                    self.client.delete_collection(col)
            # Recreate essentials
            self._ensure_collection("mem_general")
            self._ensure_collection(ENTITY_COLLECTION)
        except Exception as e:
            print(f"[Memory] Clear all error: {e}")

    def migrate_legacy(self, legacy_collections: list[str] = None,
                       project_detector=None) -> dict:
        """Migrate data from old v3/v4 collections into new project-scoped ones.
        
        Reads vectors + payload from old collections, assigns projects via detector,
        and upserts into new mem_X collections.
        """
        if legacy_collections is None:
            legacy_collections = ["memory_v4", "episodic"]
        
        migrated = {}
        try:
            all_cols = [c.name for c in self.client.get_collections().collections]
        except Exception:
            return {"error": "Cannot list collections"}
        
        for old_col in legacy_collections:
            if old_col not in all_cols:
                continue
            
            try:
                info = self.client.get_collection(old_col)
                total = info.points_count
                if total == 0:
                    continue
                
                # Scroll through all points in batches
                count = 0
                offset = None
                batch_size = 100
                
                while True:
                    results, next_offset = self.client.scroll(
                        collection_name=old_col,
                        limit=batch_size,
                        offset=offset,
                        with_vectors=True,
                        with_payload=True,
                    )
                    
                    if not results:
                        break
                    
                    # Group by project
                    project_points = {}
                    for point in results:
                        text = point.payload.get("text", "")
                        project = "general"
                        
                        # Use project detector if available
                        if project_detector and text:
                            detected = project_detector(text)
                            if detected:
                                project = detected
                        
                        # Also check existing project field
                        if point.payload.get("project"):
                            project = point.payload["project"]
                        
                        col = self._collection_name(project)
                        if col not in project_points:
                            project_points[col] = []
                        
                        new_payload = dict(point.payload)
                        new_payload["project"] = project
                        new_payload["migrated_from"] = old_col
                        
                        project_points[col].append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=point.vector,
                            payload=new_payload,
                        ))
                    
                    # Upsert to new collections
                    for col, points in project_points.items():
                        self._ensure_collection(col)
                        self.client.upsert(collection_name=col, points=points)
                        count += len(points)
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                
                migrated[old_col] = count
                print(f"[Memory] Migrated {count} points from {old_col}")
            
            except Exception as e:
                migrated[old_col] = f"error: {e}"
                print(f"[Memory] Migration error for {old_col}: {e}")
        
        return migrated

    # ─── Settings persistence (active project, etc.) ───

    SETTINGS_COLLECTION = "mem_settings"

    def _ensure_settings(self):
        """Create settings collection if needed (no vector, payload only)."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.SETTINGS_COLLECTION not in collections:
                self.client.create_collection(
                    collection_name=self.SETTINGS_COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
                self.client.create_payload_index(
                    collection_name=self.SETTINGS_COLLECTION,
                    field_name="key",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                print(f"[Memory] Created settings collection")
        except Exception as e:
            print(f"[Memory] Error ensuring settings: {e}")

    def save_setting(self, key: str, value: str):
        """Save a key-value setting in Qdrant (upsert by key)."""
        self._ensure_settings()
        try:
            # Delete existing entries with this key
            self.client.delete(
                collection_name=self.SETTINGS_COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(key="key", match=MatchValue(value=key))]
                ),
            )
            # Insert new
            dummy_vec = [0.0] * EMBED_DIM
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=dummy_vec,
                payload={"key": key, "value": value, "timestamp": time.time()},
            )
            self.client.upsert(collection_name=self.SETTINGS_COLLECTION, points=[point])
        except Exception as e:
            print(f"[Memory] Error saving setting {key}: {e}")

    def load_setting(self, key: str) -> Optional[str]:
        """Load a setting value from Qdrant. Returns None if not found."""
        self._ensure_settings()
        try:
            results, _ = self.client.scroll(
                collection_name=self.SETTINGS_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="key", match=MatchValue(value=key))]
                ),
                limit=1,
                with_payload=True,
            )
            if results:
                return results[0].payload.get("value")
        except Exception as e:
            print(f"[Memory] Error loading setting {key}: {e}")
        return None
