# We are import the require library
import faiss
import numpy as np
# We are define the class VectorDB
class VectorDB:
    def __init__(self,dimension):
        self.dimension = dimension
        self.vectors = []
        # We are create faiss index
        self.index = faiss.IndexFlatL2(dimension)
        
    def add_vector(self, vector):
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension mismatch")
        self.vectors.append(vector)
        # We are add vector to the FAISS index
        self.index.add(np.array[vector], dtype= np.float32)
        
    def search_vector(self, query_vector, k=5):
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension mismatch")
        # We are search for the nearest neighbors with FAISS
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
        return distances [0], indices[0]
    
    
# We are define example usage
if __name__ == "__main__":
    # We are create a vectorDB instance
    vector_db = VectorDB(dimension=100)
    
    # We are add vectors to the database
    vectors_to_add = [[0.1]* 100, [0.2]*100, [0.3]*100]
    for vector in vectors_to_add:
        vector_db.add_vector(vector)
        
    # We are search for nearest neighbors
    query_vector = [0.15]* 100
    distances, indices = vector_db.search_vector(query_vector)
    print("Distances:", distances)
    print("Indices:", indices)
    
