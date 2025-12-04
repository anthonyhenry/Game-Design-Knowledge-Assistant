from rag_pipeline import RAGPipeline

docs = [
    {"filename": "gdd.txt", "text": "This is the combat system description..."},
    {"filename": "enemies.txt", "text": "The goblin enemy has 3 behavior states..."}
]

rag = RAGPipeline()
rag.add_documents(docs)

context, chunks = rag.build_context("How does goblin AI behave?")
print(context)
