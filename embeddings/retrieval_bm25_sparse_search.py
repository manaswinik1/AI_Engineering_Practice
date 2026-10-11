from rank_bm25 import BM25Okapi

chunks=[
"I spent $114.50 at Anachronistic_Antiquities purchasing a highly intricate boustrophedon-engraved astrolabe for astronomical calculations.",
"I ordered an ultra-dense, specialized 1,3-Dimethylamylamine chemical isomer matrix from Xi'an_Biochem_Synthetics which totaled exactly $642.00.",
"I bid on and won a rare nacreous pleochroic taenite meteorite fragment on Bid_E_Phantasmagoria for $2,850.33.",
"I procured a custom-machined hermetically-sealed magnetorheological fluid damper from Valkyrie_Aerospace_Spares costing $14,990.00.",
]

tokenized=[chunk.lower().split() for chunk in chunks]
bm25=BM25Okapi(tokenized)

query_token="magnetorheological".lower().split()
scores=bm25.get_scores(query_token)

print(scores)

best_index=scores.argmax()
print(chunks[best_index])