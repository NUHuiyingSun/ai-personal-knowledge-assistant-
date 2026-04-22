import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP Server
mcp = FastMCP("Technical_Knowledge_Assistant")

# --- Core Fix: Obtain the absolute path of the script directory ---
# This ensures files are found regardless of where the server is launched from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(CURRENT_DIR, "notes_index.faiss")
METADATA_PATH = os.path.join(CURRENT_DIR, "notes_metadata.pkl")

print(f"Loading from: {INDEX_PATH}")

# 2. Pre-load the Model and Vector Index
# Note: The first run will download the model weights (approx. 400MB)
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Validate if the index exists before attempting to read
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"Missing index file at {INDEX_PATH}. Run build_index.py first.")

# Load the FAISS index and the corresponding metadata
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

@mcp.tool()
def search_notes(query: str) -> str:
    """
    Search local technical notes for infrastructure, 
    backend engineering, and project context.
    """
    # Convert natural language query into a vector embedding
    query_vec = model.encode([query], normalize_embeddings=True)
    
    # Perform K-Nearest Neighbors search in FAISS
    distances, indices = index.search(np.array(query_vec), k=3)
    
    # Assemble the retrieved chunks into a context string
    context = []
    for i in indices[0]:
        if i < len(metadata):
            match = metadata[i]
            context.append(f"Source: {match['file']}\nContent: {match['content']}")
    
    return "\n---\n".join(context)

if __name__ == "__main__":
    # Start the MCP server using Standard I/O (stdio) transport
    mcp.run(transport='stdio')