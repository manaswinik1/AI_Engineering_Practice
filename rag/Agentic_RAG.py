
import os
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import chromadb

load_dotenv()
model=SentenceTransformer('all-MiniLM-L6-v2')
api_key=os.environ.get("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=api_key)

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

query=input("Enter query: ")
def Consolidated_scores_and_rerank(query, top_k=5):
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

    chunk_scores={}
    chunks=fixed_size_chunks()
    for chunk, score in zip(dense_chunks, dense_score): 
        chunk_scores[chunk]=chunk_scores.get(chunk,0)+score*0.7

    for i, score in enumerate(bm25_scores):
        normalized=(score/max(bm25_scores)+1e-9)
        chunk_scores[chunks[i]]=chunk_scores.get(chunks[i], 0) + normalized*0.3

    ranked = sorted(chunk_scores.items(), key=lambda x:x[1], reverse=True)

    reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs=[(query, chunk) for chunk, score in ranked]
    rerank_scores=reranker.predict(pairs)
    ranked=sorted(zip(rerank_scores, ranked), reverse=True)
    Top5=[chunk for score, chunk in ranked[:5]]
    return Top5
   
Top5=Consolidated_scores_and_rerank(query)
for i, (chunk, score) in enumerate(Top5,1):
    print(f"{i}. {score}:{chunk}\n\n")
def extract_key_terms(query):
    stop_words={"what","when","where","why","how","did","do","does","is","are",
               "was","were","the","a","an","in","on","at","to","for","of","and",
               "i","my","me","this","that","throughout","much","many"}
    words=query.lower().replace("?", " ").split()
    key_terms=[w for w in words if w not in stop_words]
    query=" ".join(key_terms)
    return query
def evaluate_chunk_relevance(query,chunk):
    response=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        messages=[{"role": "user", "content":f"Is this chunk relevent to answering the questiion? Question: {query}, chunk: {chunk}. Answer with only: RELEVANT, PARTIAL, OR IRRELEVANT"}]
    ) 
    return response.content[0].text

ranked= Consolidated_scores_and_rerank(query)
relevance_scores=[evaluate_chunk_relevance(query, chunk) for chunk in ranked]
print(relevance_scores)
good_chunks=[c for c, r in zip(Top5, relevance_scores)
             if r in ("RELEVANT", "PARTIAL")]
print(f"Retrieved: {len(Top5)}, kept after evaluation {len(good_chunks)}")

if len(good_chunks)==0:
    print("No relevant chunks found - trying broader search")
    query=extract_key_terms(query)
    print(query)
    good_chunks=Consolidated_scores_and_rerank(query)

    if len(good_chunks)==0:
        print("Cound not find anything relevant about this in documnets")

AGENT_PROMPT=f"""Your entire response must be exactly one line, with no other text before or after it. 
Do not explain what you're about to do. 
Respond with ONLY: 
SEARCH:[query] or ANSWER:[text]


"""

def agentic_rag(query,Top5,max_steps=5):
    numbered_context=' '
    for i, chunk in enumerate(Top5):
        numbered_context+=f"[{i}] {chunk}\n\n"
    gathered=[]
    messages=[{"role":"user", "content": f"Question: {query}, Initially retrieved chunks {numbered_context} and While giving answering, Always start the answer with 'Based on [X]..., and the X is the chunk number(s) we used.Example: Based on [2] and [3]: The answer is ...'If the context doesn't contain the answer, respond with: 'This information is not available in the provided context.'Keep answers clear and concise. \nContext: None yet. What should i search for?"}]
    for step in range(max_steps):
        resp=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1234,
        system=AGENT_PROMPT,
        messages=messages
    )
        output= resp.content[0].text.strip()

        if output.startswith("ANSWER:"):
            return output[7:].strip()

        elif output.startswith("SEARCH"):
            search_query=output[7:].strip()
            print(f"Step {step+1}: Searching '{search_query}")
            retrieved = [chunk for chunk, score in Consolidated_scores_and_rerank(search_query)]


            gathered.extend(retrieved)
            messages.append({"role":"user", "content":output})
            messages.append({"role":"user", "content":"Retrived:\n" + "\n---\n".join(retrieved) + f"\nHave {len(gathered)} chunks now. enough to answer, or search more"})

        else:
            return output
    return output
output=agentic_rag(query, Top5)
print(output)