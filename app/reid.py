import os
import torch
import torchreid
import chromadb
import numpy as np
import uuid

from torchvision import transforms
from .config import CHROMADB_PATH, REID_MODEL, REID_INPUT_SHAPE, REID_SIM_THRESHOLD, REID_DISTANCE

class ReID:
    def __init__(self, db_path=CHROMADB_PATH): 
        self.reidmodel = torchreid.models.build_model(
            name=REID_MODEL,
            num_classes=1000,
            pretrained=True,
        )
        self.reidmodel.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reidmodel.to(self.device)
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(REID_INPUT_SHAPE),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        space = REID_DISTANCE
        self.collection = self.client.get_or_create_collection(
            name="objects",
            metadata={"hnsw:space": space}
        )
    
    def _get_embedding(self, image):
        assert isinstance(image, np.ndarray), "Image must be a numpy.ndarray"

        with torch.no_grad():
            preprocessed_image = self.preprocess(image)
            preprocessed_image = preprocessed_image.unsqueeze(0).to(self.device)
            
            embedding = self.reidmodel(preprocessed_image).detach().cpu().numpy()[0]
            
            # norm = np.linalg.norm(embedding) + 1e-12
            # embedding = (embedding / norm).astype(np.float32)

        return embedding

    def search(self, embedding, top_k=3, threshold: float | None = None):
        """
            Searches for similar embeddings in the database.
            Returns the ID if found, else None.
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel().tolist()
        results = self.collection.query(query_embeddings=[vec], n_results=max(1, top_k))

        ids_rows = results.get("ids", [[]])
        dist_rows = results.get("distances", [[]])
        # print(f"Search results: {ids_rows}")
        # print(f"Search results: {dist_rows}")

        if not ids_rows or not dist_rows:
            return None
        
        ids = ids_rows[0]
        dists = dist_rows[0]
        
        if not ids or not dists:
            return None

        i_min = int(np.argmin(dists))
        best_dist = float(dists[i_min])
        best_id = ids[i_min]

        metric = REID_DISTANCE
        thr = REID_SIM_THRESHOLD if threshold is None else float(threshold)

        if metric == "cosine":
            cos_sim = 1.0 - best_dist
            confidence = max(0.0, min(1.0, cos_sim))
        elif metric == "ip":
            confidence = max(0.0, min(1.0, 0.5 * (best_dist + 1.0)))
        else:
            max_l2 = np.sqrt(2.0)
            confidence = max(0.0, min(1.0, 1.0 - (best_dist / max_l2)))

        return best_id if confidence >= thr else None
    
    def append(self, embedding, metadata=None):
        """
            Appends embedding to the database.
        """
        obj_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[obj_id],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )
        
        return obj_id
        
    def add(self, image, metadata=None):
        """
            Adds an image to the database if it doesn't already exist.
            Returns the ID of the image in DB.
        """
        embedding = self._get_embedding(image)
        obj_id = self.search(embedding)
        if obj_id:
            return obj_id

        return self.append(embedding, metadata)

    def list_ids(self):
        """Return all stored IDs in the collection."""
        try:
            results = self.collection.get()
            return results.get("ids", [])
        except Exception:
            return []

    def count(self):
        """Return the number of items stored in the collection."""
        try:
            return len(self.list_ids())
        except Exception:
            return 0
