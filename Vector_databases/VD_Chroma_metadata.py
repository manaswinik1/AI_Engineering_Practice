import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
client=chromadb.PersistentClient(path="./chroma_db")

financial_data=pd.read_csv("data/finance_transactions.csv")

collection=client.get_or_create_collection(
    name="finance_transactions",
    metadata={"hnsw:space": "cosine"},
    )
model=SentenceTransformer('all-MiniLM-L6-v2')
if collection.count()==0:
    chunks=financial_data['description'].tolist()
    embeddings=model.encode(chunks).tolist()
    metadata=financial_data[['date','month','amount','merchant','category','subcategory','payment_method','transaction_type','is_recurring','need_or_want','city']].to_dict("records")
    collection.upsert(
        ids=financial_data['transaction_id'].tolist(),
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata,
    )
    print(f"Indexed {len(chunks)} chunks")
else:
    print(f"Loaded {collection.count()} existing chunks - skipping embeddings")



results=collection.query(
    query_embeddings=[model.encode("Eat out Spending").tolist()],
    n_results=3,
    where={"$and":[
        # {"merchant": {"$in": ["DoorDash"]}},
        {"month":"2025-10"},
        {"category": "Food"},
    ]}
)
for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"Score: {1-distance:.3f} | {doc}")
