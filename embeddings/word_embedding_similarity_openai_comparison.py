from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
model = SentenceTransformer('all-MiniLM-L6-v2')

profit = model.encode("profit")
revenue = model.encode("revenue")
banana = model.encode("banana")
loss = model.encode("loss")
business= model.encode("business")
secret= model.encode("secret")

print(util.cos_sim(profit, revenue))
print(util.cos_sim(profit, banana))
print(util.cos_sim(profit, loss))
print(util.cos_sim(profit, business))
print(util.cos_sim(profit, secret))

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    default_headers={"Accept-Encoding": "gzip"}
)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="profit"
)
embedding = response.data[0].embedding
print(embedding)