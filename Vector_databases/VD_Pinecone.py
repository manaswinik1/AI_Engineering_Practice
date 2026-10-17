from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

pc=Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
financial_data=pd.read_csv("data/finance_transactions.csv", keep_default_na=False)
INDEX_NAME="finance-transactions"
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

if index.describe_index_stats().total_vector_count ==0:
    chunks=financial_data['description'].tolist()
    embeddings=model.encode(chunks).tolist()
    metadata=financial_data[['date','month','amount','merchant','category','subcategory','payment_method','transaction_type','is_recurring','need_or_want','city']].to_dict("records")
    vectors=[
        (txn_id, emb, {**meta, "text":chunk})
        for txn_id, emb, meta, chunk in zip(financial_data['transaction_id'],embeddings, metadata, chunks)
    ]
    index.upsert(vectors=vectors)
    print(index.describe_index_stats())
else:
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