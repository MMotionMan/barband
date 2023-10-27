import numpy as np
import faiss


def cold_user_recommendation(user_vector: list):
    index = faiss.read_index("online_flat.index")
    user_vector = np.array(user_vector)
    D, I = index.search(np.array([user_vector]), 10)
    return list(I[0])
