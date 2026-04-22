import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownTextSplitter

# 1. Initialize the model (Will automatically download from HuggingFace on first run)
print("Loading BGE embedding model... This may take a few minutes on the first run.")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# 2. Read and split notes
notes_dir = "./my_notes"
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
metadata = []

print("Reading notes from directory...")
# Ensure the directory exists
if not os.path.exists(notes_dir):
    os.makedirs(notes_dir)
    print(f"Directory created. Please place your Markdown notes in the '{notes_dir}' folder and try again!")
    exit()

for filename in os.listdir(notes_dir):
    if filename.endswith(".md"):
        with open(os.path.join(notes_dir, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            # Split text using Markdown-aware logic
            file_chunks = splitter.split_text(text)
            for chunk in file_chunks:
                chunks.append(chunk)
                metadata.append({"file": filename, "content": chunk})

if not chunks:
    print("No Markdown files found or files are empty!")
    exit()

# 3. Convert text to vectors (Embeddings)
print(f"Converting {len(chunks)} text chunks into vectors...")
embeddings = model.encode(chunks, normalize_embeddings=True)

# 4. Store in FAISS database
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) # Using Inner Product (IP) for cosine similarity
index.add(np.array(embeddings).astype('float32'))

# 5. Save index and metadata to local disk
faiss.write_index(index, "notes_index.faiss")
with open("notes_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Knowledge base built successfully! Generated 'notes_index.faiss' and 'notes_metadata.pkl'.")