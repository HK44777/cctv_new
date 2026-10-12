from qdrant_client import QdrantClient
from qdrant_client.http import models

class DatabaseClient:
    def __init__(self, collection_name="persons", gait_dim=256, app_dim=512, face_dim=512):
        # Connect to local folder database (no Docker needed)
        self.client = QdrantClient(path="qdrant_db")
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

    def get_by_id(self, uuid: str):
        """Fetch a specific point by its UUID (used for retrieving metadata for Evidence Playback)"""
        res = self.client.retrieve(collection_name=self.collection_name, ids=[uuid], with_vectors=True)
        if res:
            return res[0]
        return None

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
