import faiss
import numpy as np
import pandas as pd


class VectorChanger:
    def __init__(self, online_index_path, query_index_path):
        self.flat_index = faiss.read_index(online_index_path)
        self.query_index = faiss.read_index(query_index_path)

    def add_to_vector(self, vector, event_id):
        try:
            self.flat_index.add_with_ids(np.array([np.array(vector)]), np.array(event_id).astype('int64'))
            self.query_index.add(np.array([np.array(vector)]))
            faiss.write_index(self.flat_index, "online_flat.index")
            faiss.write_index(self.query_index, "query.index")

        finally:
            return 0

    def delete_from_vector(self, event_id):
        try:
            self.flat_index.remove_ids(np.array([int(event_id)]).astype('int64'))
            faiss.write_index(self.flat_index, "online_flat.index")

            return 1

        finally:
            return 0

    def update_index(self):
        try:
            dim = 1024
            k = 1000
            item_vectors = np.array([self.query_index.reconstruct(i) for i in range(self.query_index.ntotal)])
            quantiser = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantiser, dim, k, faiss.METRIC_L2)
            index.train(item_vectors)

            flat_df = pd.read_csv('evants_df.csv')
            active_db = flat_df[flat_df['active'] == 1]
            indices = active_db.index.values
            active_vectors = np.array([self.query_index.reconstruct(int(i)) for i in indices])
            index.add_with_ids(active_vectors, indices)
            index.nprobe = 16
            faiss.write_index(index, "online_flat.index")

            return 1

        finally:
            return 0
