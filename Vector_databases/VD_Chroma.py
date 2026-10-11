import chromadb
from sentence_transformers import SentenceTransformer
#create client - persistent to disk so data survives restarts
client=chromadb.PersistentClient(path="./chroma_db")
#collection is like a table- holds related vectors
collection= client.get_or_create_collection(
    name="finance_docs",
    metadata={"hnsw:space": "cosine"},
    )
model=SentenceTransformer('all-MiniLM-L6-v2')



if collection.count()==0: 
    chunks=[
    "I paid $1,450 for rent on the first of the month.",
    "My electricity bill came to $112 this month because the AC ran nonstop.",
    "I spent $68 on internet service from Cox.",
    "I paid $45 for my phone plan.",
    "Groceries at Walmart cost me $87 this week.",
    "I bought organic vegetables and fruit at the farmers market for $34.",
    "I spent $22 on lunch at a Thai restaurant with a coworker.",
    "I ordered DoorDash three times this month and spent $74 in total.",
    "My Netflix subscription costs $15.49 a month.",
    "I pay $11.99 a month for Spotify Premium.",
    "I filled up my gas tank for $42 on Tuesday.",
    "My car insurance payment is $138 every month.",
    "I paid $65 for an oil change and tire rotation.",
    "I bought a monthly health insurance plan for $320.",
    "I paid $30 for a prescription refill at the pharmacy.",
    "I spent $58 on a new pair of running shoes.",
    "My gym membership is $29 a month.",
    "I bought protein powder for $39.99.",
    "I spent $19 on a haircut and $14 on a tip.",
    "I paid $24 for shampoo, conditioner, and face wash at Target.",
    "I spent $48 on skincare products, including a new serum.",
    "I paid the water and sewage bill, which was $56.",
    "I bought a $12 monthly subscription to a cloud storage service.",
    "I paid $20 a month for my ChatGPT-style AI tools subscription.",
    "I spent $9 on coffee beans from the local roastery.",
    "I got coffee at Starbucks four times and spent $21.",
    "I bought a new phone case for $17.",
    "I paid $85 for an online data science course.",
    "I spent $35 on a used book about machine learning.",
    "I paid $120 for AWS services on my personal account.",
    "I sent $200 to my parents in India.",
    "I paid $50 for a birthday gift for my friend.",
    "I spent $27 on movie tickets and popcorn.",
    "I paid $14 for parking downtown.",
    "I bought cleaning supplies and paper towels for $31.",
    "I spent $63 on a new lamp for the living room.",
    "I paid $18 for decorative pillow covers on Amazon.",
    "I paid $240 toward my student loan.",
    "I contributed $300 to my savings account this month.",
    "I spent $41 at the pharmacy on vitamins and allergy medicine.",
    "I paid $75 for a dental cleaning copay.",
    "I bought a set of spices and cooking oil for $26.",
    "I spent $92 at Costco on bulk rice, lentils, and snacks.",
    "I paid $16 for a Uber ride home after dinner.",
    "I spent $110 on a new work-from-home office chair.",
    "I paid $13 for a USB-C cable and a mouse pad.",
    "I bought a $60 gift card to treat myself at a bookstore.",
    "I paid the $9.99 monthly fee for my cloud backup service.",
    "I spent $44 on a dinner date at an Italian restaurant.",
    "I paid $33 for a late-night pizza and drinks with friends.",
    "I spent $52 on a monthly transit and toll pass.",
    "I paid $8 for a car wash on Saturday.",
    "I bought a $140 flight change fee for my trip home.",
    "I paid $25 for my annual credit card fee, split monthly.",
    "I spent $70 on makeup and nail polish.",
    "I bought a $38 jacket on sale at H&M.",
    "I paid $15 for a hair mask and scalp oil.",
    "I spent $29 on a yoga class drop-in package.",
    "I paid $72 for an electric kettle and air fryer on Black Friday.",
    "I spent $18 on stamps, envelopes, and printing at the UPS Store.",
    "I donated $25 to a local animal shelter.",
    "I paid $10 for an app subscription to track my budget.",
    "I spent $46 on snacks, drinks, and a movie night at home.",
    "I paid a $35 late fee on my utility bill because I forgot to pay on time.",
    "I bought a monthly pass for the community pool for $40."
    ]
    embeddings=model.encode(chunks).tolist()
    collection.upsert(
        ids=[f"chunks_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
    )
    print(f"Indexed {len(chunks)} chunks")
else:
    print(f"Loaded {collection.count()} existing chunks - skipping embeddings")

query="what did i spend on groceries?"
query_embedding=model.encode(query).tolist()

results=collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"Score: {1-distance/2:.3f} | {doc}")