import os
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import chromadb
model=SentenceTransformer('all-MiniLM-L6-v2')

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


def build_collection(doc_path, collection_name="Personal_Finances"):
    chunks=fixed_size_chunks()
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


def individual_scoring(query, top_k=5):
    #dense search
    query_embedding=model.encode(query).tolist()
    collection=build_collection("data/financial_report.txt")
    chunks=fixed_size_chunks()
    dense_results=collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k*2
    )
    dense_chunks=dense_results['documents'][0]
    dense_score=[1/(1+d) for d in dense_results['distances'][0]]

    #sparse search
    tokenized=[c.lower().split() for c in chunks]
    bm25=BM25Okapi(tokenized)
    bm25_scores=bm25.get_scores(query.lower().split())

    return dense_chunks,dense_score, bm25_scores
dense_chunks,dense_score, bm25_scores=individual_scoring("How much did I spend on rent throughout 2024?")


def Consolidated_scores(dense_chunks, dense_score, bm25_scores):
    chunk_scores={}
    chunks=fixed_size_chunks()
    for chunk, score in zip(dense_chunks, dense_score): 
        chunk_scores[chunk]=chunk_scores.get(chunk,0)+score*0.7

    for i, score in enumerate(bm25_scores):
        normalized=(score/max(bm25_scores)+1e-9)
        chunk_scores[chunks[i]]=chunk_scores.get(chunks[i], 0) + normalized*0.3
    return chunk_scores
chunk_scores=Consolidated_scores(dense_chunks, dense_score, bm25_scores)
print(chunk_scores)

