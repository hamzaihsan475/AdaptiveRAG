from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:


    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:

        return self.embeddings.embed_documents(texts)


if __name__ == "__main__":
    sample_chunks = [
        "This endpoint requires a Bearer token for authentication.",
        "Database failovers are handled automatically via replication.",
    ]
    embedder = EmbeddingGenerator()
    vectors = embedder.embed(sample_chunks)
    print(f"Generated {len(vectors)} embeddings, each of length {len(vectors[0])}")