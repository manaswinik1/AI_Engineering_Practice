import string


with open('../data/economic.txt', 'r') as file:
    file_content=file.read()
chunks= file_content.split("\n\n")

user_question=input("enter your question: ")
stop_words = {"what", "is", "the", "to", "according", "a", "of", "in", "on", "this", "that"}
question_words=[w for w in user_question.lower().split() if w not in stop_words]
question_words=[w.strip(string.punctuation) for w in question_words]
best_chunk=""
most_matches=0

scores = []
for chunk in chunks:
    matches = sum(1 for word in question_words if word in chunk.lower())
    scores.append(matches)

best_score = max(scores)
top_chunks = [chunks[i] for i in range(len(chunks)) if scores[i] == best_score]  
print(top_chunks) 
