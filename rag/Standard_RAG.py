import os
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import chromadb

def load_file(doc_path):
    with open (doc_path, "r") as f:
        text=f.read()
    return text
text=load_file("data/financial_report.txt")


def fixed_size_chunks(chunk_size=100, overlap=20):
    text=load_file("data/financial_report.txt")
    words=text.split()
    chunks=[]
    i=0
    while i<(len(words)): # i should be leass than len(words) if not there will be no words left to chunk
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i+=chunk_size-overlap
    return chunks
chunks=fixed_size_chunks()
print(len(chunks))

def build_collection(doc_path, collection_name="Personal_Finances"):
    chunks=fixed_size_chunks()
    model=SentenceTransformer('all-MiniLM-L6-v2')
    embeddings=model.encode(chunks).tolist()

    client=chromadb.PersistentClient('./chroma_db')
    collection=client.get_or_create_collection(collection_name)

    if collection.count() ==0:
        collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"Chunk_index": i, "source":doc_path}  
                       for i in range(len(chunks))]
        )
        print(f"Indexed {len(chunks)} chunks")
    else:
        print(f"Using existing index: {collection.count()} chunks")
    return collection
collection=build_collection("data/financial_report.txt")
print(collection.count())