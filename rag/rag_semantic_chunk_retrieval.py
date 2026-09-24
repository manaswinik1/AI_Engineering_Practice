import string
from sentence_transformers import SentenceTransformer, util
import torch
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('../data/economic.txt', 'r') as file:
    file_content=file.read()
chunks= file_content.split("\n\n")
chunk_embeddings=model.encode(chunks)
user_question=input("enter the question: ")
question_embedding=model.encode(user_question)

similarity_question_chunk=util.cos_sim(chunk_embeddings,question_embedding)
global_idx = torch.argmax(similarity_question_chunk)
print(global_idx)
best_chunk=chunks[global_idx]
print(best_chunk)