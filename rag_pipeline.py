import numpy as np
from sentence_transformers import SentenceTransformer
import os

class RAGPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.documents = []
        self.chunks = []
        self.chunk_sources = []
        self.embeddings = None
        
        # Load local embedding model (first download takes ~50MB)
        self.model = SentenceTransformer(model_name)

    # ------------------------
    # 1. Add documents
    # ------------------------
    def add_documents(self, docs):
        self.documents = docs
        self._create_chunks()
        self._embed_chunks()

    # ------------------------
    # 2. Chunk document text
    # ------------------------
    def _create_chunks(self, chunk_size=600, overlap=100):
        all_chunks = []
        all_sources = []

        for doc in self.documents:
            text = doc["text"]
            words = text.split()

            i = 0
            while i < len(words):
                chunk_words = words[i: i + chunk_size]
                chunk_text = " ".join(chunk_words)

                all_chunks.append(chunk_text)
                all_sources.append(doc["filename"])

                i += chunk_size - overlap

        self.chunks = all_chunks
        self.chunk_sources = all_sources

    # ------------------------
    # 3. Embed chunks (FREE, local)
    # ------------------------
    def _embed_chunks(self):
        if not self.chunks:
            self.embeddings = None
            return

        # Convert all chunks into vector embeddings
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)

    # ------------------------
    # 4. Similarity search
    # ------------------------
    def search(self, query, top_k=4):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Encode query using same model
        q_vec = self.model.encode(query, convert_to_numpy=True)

        # Compute cosine similarities
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_vec)
        similarities = np.dot(self.embeddings, q_vec) / norms

        top_idx = similarities.argsort()[-top_k:][::-1]

        results = [
            {
                "chunk": self.chunks[i],
                "source": self.chunk_sources[i],
                "score": float(similarities[i])
            }
            for i in top_idx
        ]

        MIN_SIMILARITY = 0.30   # This can be adjusted
        results = [r for r in results if r["score"] >= MIN_SIMILARITY]

        return results


    # ------------------------
    # 5. Build context
    # ------------------------
    def build_context(self, query, top_k=4):
        retrieved = self.search(query, top_k)

        context = ""
        for r in retrieved:
            context += f"[From: {r['source']}]\n{r['chunk']}\n\n"

        return context, retrieved



# import numpy as np
# from openai import OpenAI
# import os

# # Get api key from .env file
# from dotenv import load_dotenv
# load_dotenv()
# # Get api key from .env file


# class RAGPipeline:
#     def __init__(self):
#         self.documents = []         # list of {"filename":..., "text":...}
#         self.chunks = []            # list of chunk strings
#         self.chunk_sources = []     # which document each chunk came from
#         self.embeddings = None      # numpy matrix of vectors
#         api_key = os.getenv("OPENAI_API_KEY")
#         self.client = OpenAI(api_key=api_key)

#     # ------------------------
#     # 1. Add documents
#     # ------------------------
#     def add_documents(self, docs):
#         """
#         docs = list of { "filename": str, "text": str }
#         """
#         self.documents = docs
#         self._create_chunks()
#         self._embed_chunks()

#     # ------------------------
#     # 2. Chunk document text
#     # ------------------------
#     def _create_chunks(self, chunk_size=600, overlap=100):
#         """
#         Basic sliding window chunking.
#         Good balance between context and speed.
#         """
#         all_chunks = []
#         all_sources = []

#         for doc in self.documents:
#             text = doc["text"]
#             words = text.split()

#             i = 0
#             while i < len(words):
#                 chunk_words = words[i : i + chunk_size]
#                 chunk_text = " ".join(chunk_words)

#                 all_chunks.append(chunk_text)
#                 all_sources.append(doc["filename"])

#                 i += chunk_size - overlap  # sliding window

#         self.chunks = all_chunks
#         self.chunk_sources = all_sources

#     # ------------------------
#     # 3. Create embeddings
#     # ------------------------
#     def _embed_chunks(self):
#         if not self.chunks:
#             self.embeddings = None
#             return

#         response = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=self.chunks
#         )

#         vectors = [e.embedding for e in response.data]
#         self.embeddings = np.array(vectors)

#     # ------------------------
#     # 4. Similarity search
#     # ------------------------
#     def search(self, query, top_k=4):
#         """
#         Returns the top_k most relevant chunks.
#         """
#         if self.embeddings is None or len(self.chunks) == 0:
#             return []

#         # Compute embedding for query
#         q_embed = self.client.embeddings.create(
#             model="text-embedding-3-small",
#             input=query
#         ).data[0].embedding

#         q_vec = np.array(q_embed)

#         # Cosine similarity
#         norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_vec)
#         similarities = np.dot(self.embeddings, q_vec) / norms

#         # Get top ranked
#         top_idx = similarities.argsort()[-top_k:][::-1]

#         results = [
#             {
#                 "chunk": self.chunks[i],
#                 "source": self.chunk_sources[i],
#                 "score": float(similarities[i])
#             }
#             for i in top_idx
#         ]

#         return results

#     # ------------------------
#     # 5. Build context to send to LLM
#     # ------------------------
#     def build_context(self, query, top_k=4):
#         retrieved = self.search(query, top_k)

#         context = ""
#         for r in retrieved:
#             context += f"[From: {r['source']}]\n{r['chunk']}\n\n"

#         return context, retrieved
