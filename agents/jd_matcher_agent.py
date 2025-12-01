from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
import numpy as np

class JDMatcherAgent:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")

    def match(self, cv_chunks, job_description):
        cv_embeddings = self.embedder.embed_documents(cv_chunks)
        jd_embedding = self.embedder.embed_query(job_description)

        similarities = [cosine_similarity([jd_embedding], [vec])[0][0] for vec in cv_embeddings]
        max_score = max(similarities)
        avg_score = float(np.mean(similarities))

        return {
            "similarity_scores": similarities,
            "max_score": max_score,
            "avg_score": avg_score
        }