import chromadb
from sentence_transformers import SentenceTransformer
import os
from rank_bm25 import BM25Okapi
def fixed_size_chunks(text, chunk_size=100, overlap=20):
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
    return text, collection, model

text, collection, model= build_index("data/comprehensive_science.txt")
chunks=fixed_size_chunks(text)
#dense retrival
top_k=5
query="dopamine and motivation"
query_embedding=model.encode(query).tolist()
dense_results=collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k*2
)
dense_chunks=dense_results['documents'][0]
dense_scores=[1/(1+d) for d in dense_results['distances'][0]]

#BM25 retrival
tokenized = [c.lower().split() for c in chunks]
bm25=BM25Okapi(tokenized)
bm25_scores = bm25.get_scores(query.lower().split())


#combine weights dense 70%, BM25 30%
chunk_score={}
for chunk, score in zip(dense_chunks, dense_scores):
    chunk_score[chunk]=chunk_score.get(chunk, 0)+ score*0.7

for i, score in enumerate(bm25_scores):
    normalized= score/(max(bm25_scores) + 1e-9)
    chunk_score[chunks[i]]=chunk_score.get(chunks[i], 0) + normalized*0.3

ranked= sorted(chunk_score.items(), key=lambda x:x[1], reverse=True)

print(chunk_score.items())