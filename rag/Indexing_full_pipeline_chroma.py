import chromadb
from sentence_transformers import SentenceTransformer
import os

def fixed_size_chunks(text, chunk_size=50, overlap=5):
    words=text.split()
    chunks=[]
    i=0
    while i<(len(words)):
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i+=chunk_size-overlap
    return chunks

def build_index(doc_path, collection_name="rag_docs"):
    with open(doc_path, "r") as f:
        text=f.read()
    chunks=fixed_size_chunks(text)

    model=SentenceTransformer('all-MiniLM-L6-v2')
    embeddings=model.encode(chunks).tolist()

    client=chromadb.PersistentClient(path="./chroma_db")
    collection=client.get_or_create_collection(collection_name)

    if collection.count()==0:
        collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"chunk_index": i, "source": doc_path}
                      for i in range(len(chunks))]
        )
        print(f"Indexed {len(chunks)} chunks")
    else:
        print(f"Using existing index: {collection.count()} chunks")
    return collection, model

collection, model= build_index("data/book_excerpt.txt")
