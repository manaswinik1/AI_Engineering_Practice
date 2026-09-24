# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding = model.encode("profit")
# # print(embedding)
# print(len(embedding))


from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

word1 = model.encode("profit")
word2 = model.encode("cost")
word3 = model.encode("banana")

similarity_profit_cost = util.cos_sim(word1, word2)
similarity_profit_banana = util.cos_sim(word1, word3)

print("profit vs cost:", similarity_profit_cost)
print("profit vs banana:", similarity_profit_banana)