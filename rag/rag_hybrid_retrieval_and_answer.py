from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import anthropic
import torch
import string 
load_dotenv()
model=SentenceTransformer('all-MiniLM-L6-v2')
api_key=os.environ.get("ANTHROPIC_API_KEY")
client=anthropic.Anthropic(api_key=api_key)
with open('../data/economic.txt','r') as file:
    file_content=file.read()
chunks=file_content.split("\n\n")
embedded_chunks=model.encode(chunks)
user_question=input('enter question: ')
# keyword score for each chunk


best_idx = scores.index(max(scores))
best_chunk = chunks[best_idx]
embedded_question=model.encode(user_question)
similarity_btw_chunk_question= util.cos_sim(embedded_chunks,embedded_question)
global_idx=torch.argmax(similarity_btw_chunk_question)

scores = []
question_words = [w.strip(string.punctuation).lower() 
                  for w in user_question.split() 
                  if w.lower() not in {"what","is","the","a","of","in","to"}]

for i, chunk in enumerate(chunks):
    keyword_score = sum(1 for w in question_words if w in chunk.lower())
    semantic_score = similarity_btw_chunk_question[i].item()
    combined = (keyword_score * 0.3) + (semantic_score * 0.7)
    scores.append(combined)
    best_idx = scores.index(max(scores))  # scores is still empty []
    best_chunk = chunks[best_idx] 
message=client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1234,
    messages=[
        {"role":"user","content":f"hey claude! {user_question} from this text {chunks[global_idx]}."}
    ]
)
print(message.content[0].text)