from typing import Any, List, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models

class DatabaseClient:
    _shared_client = None

    def __init__(self, collection_name="persons", gait_dim=256, app_dim=512, face_dim=512):
        # Connect to local folder database (reuse existing connection if already opened in this process)
        if DatabaseClient._shared_client is None:
            DatabaseClient._shared_client = QdrantClient(path="qdrant_db")
        self.client = DatabaseClient._shared_client
        self.collection_name = collection_name
        self.gait_dim = gait_dim
        self.app_dim = app_dim
        self.face_dim = face_dim
        self._ensure_collection()
        print(f"Connected to Qdrant Database (collection: {self.collection_name}, gait_dim: {self.gait_dim})")

    def _ensure_collection(self):
        """
        Creates the Qdrant collection if it doesn't exist.
        We use Named Vectors for Late Fusion (App, Face, Gait).
        """
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "appearance": models.VectorParams(size=self.app_dim, distance=models.Distance.COSINE),
                    "face": models.VectorParams(size=self.face_dim, distance=models.Distance.COSINE),
                    "gait": models.VectorParams(size=self.gait_dim, distance=models.Distance.COSINE)
                }
            )

    def clear_collection(self):
        """Wipes the collection so it can be cleanly re-ingested with fresh multi-modal vectors."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()
        print(f"Collection '{self.collection_name}' cleared and re-initialized.")

    def close(self):
        """Explicitly closes the local Qdrant database to prevent shutdown warnings."""
        if DatabaseClient._shared_client is not None:
            try:
                DatabaseClient._shared_client.close()
            except Exception:
                pass
            DatabaseClient._shared_client = None

    def get_by_id(self, uuid: str):
        """Fetch a specific point by its UUID (used for retrieving metadata for Evidence Playback)"""
        res = self.client.retrieve(collection_name=self.collection_name, ids=[uuid], with_vectors=True)
        if res:
            return res[0]
        return None

    def get_by_camera_and_track(self, cam_id: str, track_id: Any) -> list:
        """
        Finds all Qdrant points that match a specific camera_id AND track_id
        using a payload filter — no vector search is performed.

        Supports both prefixed (e.g. 'Camera_1_5') and unprefixed (e.g. '5' or 5)
        track IDs seamlessly.
        """
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

        tid_str = str(track_id).strip()
        candidates = [tid_str]

        # Generate alternate formats
        if not tid_str.startswith(f"{cam_id}_"):
            candidates.append(f"{cam_id}_{tid_str}")
        else:
            candidates.append(tid_str[len(cam_id) + 1:])

        try:
            candidates.append(int(tid_str.split("_")[-1]))
        except ValueError:
            pass

        # Try MatchAny first
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="camera_id", match=MatchValue(value=cam_id)),
                        FieldCondition(key="track_id",  match=MatchAny(any=candidates)),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=15,
            )
            if results:
                return results
        except Exception:
            pass

        # Fallback: sequential MatchValue tests
        for cand in candidates:
            try:
                results, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="camera_id", match=MatchValue(value=cam_id)),
                            FieldCondition(key="track_id",  match=MatchValue(value=cand)),
                        ]
                    ),
                    with_payload=True,
                    with_vectors=False,
                    limit=15,
                )
                if results:
                    return results
            except Exception:
                continue

        return []


    def _align_vector(self, vec, expected_dim):
        """Safely aligns vector length to expected dimension to prevent Qdrant errors."""
        if vec is None:
            return None
        v = list(vec)
        if len(v) == expected_dim:
            return v
        if len(v) < expected_dim:
            return v + [0.0] * (expected_dim - len(v))
        return v[:expected_dim]

    def insert(self, uuid, appearance, face, gait, metadata):
        """
        Step 11: Database Insertion (Late Fusion / Named Vectors)
        """
        self.insert_batch([{
            "uuid": uuid,
            "appearance": appearance,
            "face": face,
            "gait": gait,
            "metadata": metadata
        }])

    def insert_batch(self, items):
        """
        Batch Database Insertion for high throughput.
        items: list of dicts with keys: uuid, appearance, face, gait, metadata
        """
        if not items:
            return
        points = []
        for it in items:
            vectors = {}
            if it.get("appearance") is not None:
                vectors["appearance"] = self._align_vector(it["appearance"], self.app_dim)
            if it.get("face") is not None:
                vectors["face"] = self._align_vector(it["face"], self.face_dim)
            if it.get("gait") is not None:
                vectors["gait"] = self._align_vector(it["gait"], self.gait_dim)

            points.append(
                models.PointStruct(
                    id=it["uuid"],
                    vector=vectors,
                    payload=it["metadata"]
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search_ensemble(self, query_appearance=None, query_face=None, query_gait=None, limit=20):
        """
        Step 14: Ensemble Database Search (Late Fusion)
        Perform independent searches for available modalities.
        """
        results = {"appearance": [], "face": [], "gait": []}
        
        if query_appearance is not None:
            res_app = self.client.query_points(
                collection_name=self.collection_name,
                query=self._align_vector(query_appearance, self.app_dim),
                using="appearance",
                limit=limit
            )
            results["appearance"] = res_app.points
            
        if query_face is not None:
            res_face = self.client.query_points(
                collection_name=self.collection_name,
                query=self._align_vector(query_face, self.face_dim),
                using="face",
                limit=limit
            )
            results["face"] = res_face.points
            
        if query_gait is not None:
            res_gait = self.client.query_points(
                collection_name=self.collection_name,
                query=self._align_vector(query_gait, self.gait_dim),
                using="gait",
                limit=limit
            )
            results["gait"] = res_gait.points
            
        return results
