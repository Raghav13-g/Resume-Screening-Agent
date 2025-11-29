from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class ResumeVectorStore:
    def __init__(self, path="./chroma_storage", collection_name="resumes"):
        self.client = PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(name=collection_name)

    def reset_collection(self):
        try:
            self.col.delete(where={})
        except:
            pass

    def embed(self, text):
        return model.encode(text, normalize_embeddings=True).tolist()

    def add_resume(self, doc_id, text, metadata):
        emb = self.embed(text)
        self.col.add(ids=[doc_id], documents=[text], metadatas=[metadata], embeddings=[emb])

    def query(self, text, n_results=10):
        q_emb = self.embed(text)
        res = self.col.query(query_embeddings=[q_emb], n_results=n_results, include=["documents","metadatas","distances"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"document": doc, "metadata": meta, "distance": dist})
        return out
