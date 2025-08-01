import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict

# Global variables
vector_index = None
vector_texts = []
vector_metadatas = []
embedding_model = None
dimension = None

def initialize_embeddings():
    """Initialize the embedding model"""
    global embedding_model, dimension
    print("Initializing embedding model")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding model loaded. Dimension: {dimension}")
    return embedding_model

def create_vector_index():
    """Create FAISS vector index"""
    global vector_index, dimension
    if dimension is None:
        initialize_embeddings()
    
    print("Creating FAISS vector index")
    vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    print("FAISS index created successfully")
    return vector_index

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for texts"""
    global embedding_model
    
    if embedding_model is None:
        initialize_embeddings()
    
    print(f"Generating embeddings for {len(texts)} texts")
    
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embedding_model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        all_embeddings.extend(embeddings)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Generated embeddings for {i + len(batch)} texts")
    
    embeddings_array = np.array(all_embeddings, dtype='float32')
    print(f"Generated embeddings with shape: {embeddings_array.shape}")
    return embeddings_array

def add_to_vector_database(chunks: List[Dict]):
    """Add processed chunks to vector database"""
    global vector_index, vector_texts, vector_metadatas
    
    print("Adding chunks to vector database")
    
    if vector_index is None:
        create_vector_index()
    
    # Extract texts and metadata
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Add to FAISS index
    vector_index.add(embeddings)
    
    # Store texts and metadata
    vector_texts.extend(texts)
    vector_metadatas.extend(metadatas)
    
    print(f"Added {len(chunks)} chunks to vector database")
    print(f"Total chunks in database: {len(vector_texts)}")

def similarity_search(query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
    """Search for similar chunks in vector database"""
    global vector_index, vector_texts, vector_metadatas, embedding_model
    
    if vector_index is None or vector_index.ntotal == 0:
        print("Vector database is empty")
        return []
    
    if embedding_model is None:
        initialize_embeddings()
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype='float32')
    faiss.normalize_L2(query_embedding)
    
    # Search for similar chunks
    scores, indices = vector_index.search(query_embedding, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > 0.3:  # Relevance threshold
            results.append((
                vector_texts[idx],
                float(score),
                vector_metadatas[idx]
            ))
    
    print(f"Found {len(results)} relevant chunks for query")
    return results

def save_vector_database(path: str):
    """Save vector database to disk"""
    global vector_index, vector_texts, vector_metadatas, dimension
    
    print("Saving vector database")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save FAISS index
    if vector_index and vector_index.ntotal > 0:
        faiss.write_index(vector_index, f"{path}.faiss")
    
    # Save texts and metadata
    with open(f"{path}.pkl", 'wb') as f:
        pickle.dump({
            'texts': vector_texts,
            'metadatas': vector_metadatas,
            'dimension': dimension
        }, f)
    
    print(f"Vector database saved to {path}")

def load_vector_database(path: str) -> bool:
    """Load vector database from disk"""
    global vector_index, vector_texts, vector_metadatas, dimension
    
    try:
        print("Loading vector database...")
        
        # Load FAISS index
        if os.path.exists(f"{path}.faiss"):
            vector_index = faiss.read_index(f"{path}.faiss")
        
        # Load texts and metadata
        if os.path.exists(f"{path}.pkl"):
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
                vector_texts = data['texts']
                vector_metadatas = data['metadatas']
                dimension = data['dimension']
        
        # Initialize embedding model
        initialize_embeddings()
        
        print(f"Loaded vector database with {len(vector_texts)} chunks")
        return True
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return False

def get_database_stats() -> Dict:
    """Get vector database statistics"""
    global vector_index, vector_texts
    return {
        'total_chunks': len(vector_texts),
        'index_size': vector_index.ntotal if vector_index else 0,
        'dimension': dimension,
        'model': 'all-MiniLM-L6-v2'
    }
