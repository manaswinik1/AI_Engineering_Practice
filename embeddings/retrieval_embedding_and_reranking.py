chunks = [
    "3BHK Interior Design Plan — Apartment #414",
    "Overall Palette: Colors: Off-white (walls, ceiling, flooring — dominant, ~70%) + Black (furniture, cupboards, doors, accents) + Gold (hardware, lighting fixtures, accents)",
    "Same palette applied consistently across every room — no room-by-room color switching",
    "Texture variety on black elements: cupboards = glossy black; furniture (sofa, etc.) = mixed textures like velvet/matte — not everything shiny",
    "Lighting: warm-toned throughout, no cool white or harsh focused white light",
    "Living Room Layout: Main door at bottom-left corner, TV wall on short wall beside the main door, Prayer arch wall directly opposite the main door (shared wall with Guest Bedroom), Sofa faces TV wall",
    "Prayer Wall: Black arched door-frame/arc built onto off-white wall, centered with centerpiece frame (Jesus/Bible) mounted inside arch",
    "Prayer Wall Storage: Floating black shelf below frame, Black cabinet at the bottom for storing religious books/artifacts",
    "Prayer Wall Lighting: Small warm spotlight at top of arc, narrow beam (15°–24°), angled down onto the frame for a focused oval halo effect (church-style)",
    "Prayer Wall Finish: No extra wall texture — kept plain to let the arch stand out",
    "TV Wall: Wall has an inward recess/niche where the 55\" TV sits with a subtle off-white texture (light random/pebble texture) on the back wall",
    "TV Wall Lighting: LED strip lighting on 3 sides of niche (top + two sides) — on its own separate switch/remote from the prayer wall light",
    "TV Wall Mount: TV mounted on pull-out/articulating panel — fully retracts flat into niche (niche must be built deep enough for this)",
    "TV Wall Cable Management: Concealed compartment behind TV for wires/sockets — nothing visible",
    "TV Cabinet: Long TV cabinet below niche, TV width + 1 ft extra on each side",
    "TV Cabinet Compartments: One concealed compartment for Wi-Fi router, sockets and one open section for books, papers, storage",
    "Sofa Wall Seating: 3-seater sofa (center) + 2 single sofas (either side)",
    "Sofa Wall Accents: Gold plant stands on either side of sofa set, with snake plants (or similar indoor plants)",
    "Sofa Wall Art: One black-framed art piece above sofa — wide, shorter height, slightly narrower than sofa width; minimal/abstract art with white or gold tones to echo palette",
    "Ceiling Structure: Flat ceiling, no false ceiling/drop structure",
    "Ceiling Fan: Chandelier fan (black + gold finish) centered near sofa seating area — acts as ceiling centerpiece",
    "Ceiling Lighting: Ring of 4–6 recessed warm downlights spaced ~3–4 ft apart around the fan, plus 2 additional recessed lights pulled toward TV wall side for coverage",
    "Ceiling Lighting Note: No ceiling light needed directly over prayer wall (handled by dedicated spotlight)",
    "Dining Area Furniture: Black dining table, matte/semi-matte finish (not glossy — avoids competing with kitchen cupboards)",
    "Dining Area Chairs: Mixed texture (e.g., black velvet/leather-look seats, black or gold slim metal legs)",
    "Dining Area Lighting: Warm-toned pendant light or cluster of 2–3 small pendants over table, gold fixture finish",
    "Dining Area Accent: Gold-rimmed centerpiece bowl or small gold candle stand",
    "Dining Wall Combo: Crockery + Photo Wall combo on the wall bordering the toilets",
    "Dining Storage & Art: Bottom features a wood-finish crockery cabinet of human height (sophisticated wood look — intentional break from black/glossy) with a photo wall of black frames above",
    "Dining Space Constraints: Utility door/window wall kept clear (no furniture placed there) and open kitchen side kept clear (no cabinet blocking sightline)",
    "Kitchen status: To be discussed",
    "Bedrooms (Master, Guest, Common) status: To be discussed",
    "Open Questions: Exact chandelier fan model/style, Dining pendant fixture selection, Kitchen layout and cabinetry details, Bedroom-specific plans"
]


from sentence_transformers import SentenceTransformer, CrossEncoder, util
#stage1
embedding_model= SentenceTransformer('all-MiniLM-L6-v2')
chunks_embeddings=embedding_model.encode(chunks)

query="what is the colour pallatte we decided on?"
query_embedding=embedding_model.encode(query)
scores=util.cos_sim(chunks_embeddings, query_embedding)
top10_indices=scores.argsort(descending=True,axis=0).flatten().tolist()[:10]
top10_chunks=[chunks[i] for i in top10_indices]
for rank, chunk in enumerate(top10_chunks, 1):
    print(f"Rank {rank}: {chunk}")

#stage2
reranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
pairs=[(query,chunk) for chunk in top10_chunks]
rerank_scores=reranker.predict(pairs)

ranked=sorted(zip(rerank_scores, top10_chunks), reverse=True)
top3=[chunk for score, chunk in ranked[:3]]
for rank, chunk in enumerate(top3, 1):
    print(f"Rank {rank}: {chunk}")