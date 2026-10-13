from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

pc=Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
financial_data2526=pd.read_csv("data/finance_transactions.csv", keep_default_na=False)
financial_data24=pd.read_csv("data/finance_transactions_2024.csv", keep_default_na=False)
INDEX_NAME="finance-transactions-24-26"
DIMENSION=384
model=SentenceTransformer('all-MiniLM-L6-v2')
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index {INDEX_NAME} Created")
else: 
    print(f"Index {INDEX_NAME} Present")

index=pc.Index(INDEX_NAME)


chunks24=chunks=financial_data24['description'].tolist()
embeddings24=model.encode(chunks).tolist()
metadata24=financial_data24[['date','month','amount','merchant','category','subcategory','payment_method','transaction_type','is_recurring','need_or_want','city']].to_dict("records")
vectors24=[
    (txn_id, emb, {**meta, "text":chunk})
    for txn_id, emb, meta, chunk in zip(financial_data24['transaction_id'],embeddings24, metadata24, chunks24)
]
chunks2526=chunks=financial_data2526['description'].tolist()
embeddings2526=model.encode(chunks2526).tolist()
metadata2526=financial_data2526[['date','month','amount','merchant','category','subcategory','payment_method','transaction_type','is_recurring','need_or_want','city']].to_dict("records")
vectors2526=[
    (txn_id, emb, {**meta, "text":chunk})
    for txn_id, emb, meta, chunk in zip(financial_data2526['transaction_id'],embeddings2526, metadata2526, chunks2526)
]

index.upsert(vectors=vectors24, namespace="2024")
index.upsert(vectors=vectors2526, namespace="2025-2026")
print(index.describe_index_stats())

print("Index populated")

query="grocery spending"
query_embedding=model.encode(query).tolist()
results=index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']:.3f} | {match['metadata']['text']}")