from qdrant_client import QdrantClient
from qdrant_client.http import models

class DatabaseClient:
    def __init__(self):
        # Connect to local folder database (no Docker needed)
        self.client = QdrantClient(path="qdrant_db")
        self.collection_name = "persons"
        self._ensure_collection()
        print("Connected to Qdrant Database")

    def _ensure_collection(self):
        """
        Creates the Qdrant collection if it doesn't exist.
        We use Named Vectors for Late Fusion (App: 512, Face: 512, Gait: 256).
        """
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "appearance": models.VectorParams(size=512, distance=models.Distance.COSINE),
                    "face": models.VectorParams(size=512, distance=models.Distance.COSINE),
                    "gait": models.VectorParams(size=256, distance=models.Distance.COSINE)
                }
            )

    def get_by_id(self, uuid: str):
        """Fetch a specific point by its UUID (used for retrieving metadata for Evidence Playback)"""
        res = self.client.retrieve(collection_name=self.collection_name, ids=[uuid], with_vectors=True)
        if res:
            return res[0]
        return None

    def insert(self, uuid, appearance, face, gait, metadata):
        """
        Step 11: Database Insertion (Late Fusion / Named Vectors)
        """
        vectors = {}
        if appearance is not None:
            vectors["appearance"] = appearance
        if face is not None:
            vectors["face"] = face
        if gait is not None:
            vectors["gait"] = gait

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=uuid,
                    vector=vectors,
                    payload=metadata
                )
            ]
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
                query=query_appearance,
                using="appearance",
                limit=limit
            )
            results["appearance"] = res_app.points
            
        if query_face is not None:
            res_face = self.client.query_points(
                collection_name=self.collection_name,
                query=query_face,
                using="face",
                limit=limit
            )
            results["face"] = res_face.points
            
        if query_gait is not None:
            res_gait = self.client.query_points(
                collection_name=self.collection_name,
                query=query_gait,
                using="gait",
                limit=limit
            )
            results["gait"] = res_gait.points
            
        return results
