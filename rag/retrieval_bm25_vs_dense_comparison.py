from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks=[
"I bought a cheap wireless mouse for $4 at AMZN MRKT to fix my broken office setup",
"I ordered a warm winter hoodie for $15 from Target for my morning runs.",
"I picked up a small desk plant for $5 at Walmart to add color to my room.",
"I found an old comic book for $8 on eBay to complete my childhood collection.",
]


embeddings=model.encode([chunk.lower() for chunk in chunks])
drugs = model.encode("jogging")
simitalrity=util.cos_sim(drugs,embeddings)
best_index=simitalrity.argmax()
print(simitalrity)
print(f"dense {chunks[best_index]}")


tokenized=[chunk.lower().split() for chunk in chunks]
bm25=BM25Okapi(tokenized)
query_token="jogging".lower().split()
scores=bm25.get_scores(query_token)
print(scores)
best_index=scores.argmax()
print(f"Sparce: {chunks[best_index]}")

