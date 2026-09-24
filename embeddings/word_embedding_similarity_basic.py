from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

embedding = model.encode("profit")
print(len(embedding))
print(embedding)

#Proof Meaning is captured

profit = model.encode("profit")
revenue = model.encode("revenue")
banana = model.encode("banana")

print(util.cos_sim(profit, revenue))
print(util.cos_sim(profit, banana))



