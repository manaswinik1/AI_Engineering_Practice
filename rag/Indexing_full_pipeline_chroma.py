import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from rank_bm25 import BM25Okapi
import anthropic
from dotenv import load_dotenv

#CHUNKING
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
#RETRIVAL
text, collection, model= build_index("data/comprehensive_science.txt")
chunks=fixed_size_chunks(text)
#dense retrival
def hybrid_score_and_rank(dense_chunks, dense_scores, chunks, bm25_scores, top_k=5, dense_weight=0.7, bm25_weight=0.3):
    """
    Combine dense and BM25 scores and rank results.

    Args:
        dense_chunks: chunks from dense retrieval
        dense_scores: similarity scores from dense retrieval
        chunks: all chunks in the corpus
        bm25_scores: keyword relevance scores from BM25
        top_k: number of top results to return
        dense_weight: weight for dense scores (default 0.7)
        bm25_weight: weight for BM25 scores (default 0.3)

    Returns:
        List of top k chunks ranked by combined score
    """
    chunk_score = {}

    # Add dense scores (70% weight)
    for chunk, score in zip(dense_chunks, dense_scores):
        chunk_score[chunk] = chunk_score.get(chunk, 0) + score * dense_weight

    # Add BM25 scores (30% weight)
    for i, score in enumerate(bm25_scores):
        normalized = score / (max(bm25_scores) + 1e-9)
        chunk_score[chunks[i]] = chunk_score.get(chunks[i], 0) + normalized * bm25_weight

    # Sort by score (highest first) and return top k
    ranked = sorted(chunk_score.items(), key=lambda x: x[1], reverse=True)
    return [(chunk, score) for chunk, score in ranked[:top_k]]


# Usage
top_k = 5
query = "Name the two main brain regions involved in reward processing and dopamine."
query_embedding = model.encode(query).tolist()

# Dense retrieval
dense_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k*2
)
dense_chunks = dense_results['documents'][0]
dense_scores = [1/(1+d) for d in dense_results['distances'][0]]

# BM25 retrieval
tokenized = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)
bm25_scores = bm25.get_scores(query.lower().split())

# Get ranked results
results = hybrid_score_and_rank(dense_chunks, dense_scores, chunks, bm25_scores, top_k=5)

#RERANKING
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs=[(query, chunk) for chunk, score in results]

rerank_scores=reranker.predict(pairs)

ranked=sorted(zip(rerank_scores, results), reverse=True)
top3=[chunk for score, chunk in ranked[:3]]
for i, (chunk, score) in enumerate(top3, 1):
    print(f"{i}. {score}: {chunk}")

#AGUMENTATION
SYSTEM_PROMPT="""You are a document-based QA assistant. 
Answer the user's question using ONLY the provided context.
Do not use external knowledge.
Always start the answer with "Based on [X]..., and the X is the chunk number(s) we used.
Example: Based on [2] and [3]: The answer is ..."
If the context doesn't contain the answer, respond with: "This information is not available in the provided context."
Keep answers clear and concise.
"""
#GENERATION
def generate_answer_with_citations(query, top3):
    numbered_context=" "
    for i, chunk in enumerate(top3):
        numbered_context+=f"[{i}] {chunk}\n\n"
    load_dotenv()
    api_key=os.environ.get("ANTHROPIC_API_KEY")
    client=anthropic.Anthropic(api_key=api_key)
    message=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content":f"Context: \n{numbered_context} \n\n Question:{query}"}]
    )
    return message.content[0].text
response=generate_answer_with_citations(query, top3) 

print(f"Result of the RAG pipeline: {response}")