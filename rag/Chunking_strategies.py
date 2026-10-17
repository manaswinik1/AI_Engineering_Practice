with open('data/book_excerpt.txt' , 'r') as file:
    file_content=file.read()

# Strategy1 Paragraph Split
chunks=file_content.split("\n\n")
chunks=[c.strip() for c in chunks if c.strip()]
print(chunks)


# Strategy2: Fixed-chunk split

def fixed_size_chunks(file_content, chunk_size=50, overlap=5):
    words=file_content.split()
    chunks=[]
    i=0
    while i < len(words):
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i+=chunk_size-overlap  # Move forward by chunk_size minus overlap
    return chunks

chunks = fixed_size_chunks(file_content)
print(chunks)


#Strategy3: Recursive Split

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
chunks=splitter.split_text(file_content)
print(chunks)
