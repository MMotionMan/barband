import faiss
import numpy as np
import pandas as pd
from navec import Navec
import navec

'''
Create an index
'''


# This function add and remove new data to index or updte index
# event_id:int
# vector:np.array
# add_delete_update:int
def add_update_delete_index(event_id=None, vector=None, add_delete_update=0):
    # Read existance index
    flat_index = faiss.read_index("./online_flat.index")
    query_index = faiss.read_index("./query.index")
    # add_delete_update == 0 - add index
    # add_delete_update == 1 - delete index
    # add_delete_update == 2 - update index
    if add_delete_update == 0:
        flat_index.add_with_ids(np.array([np.array(vector)]), np.array(event_id).astype('int64'))
        query_index.add(np.array([np.array(vector)]))
        faiss.write_index(flat_index, "online_flat.index")
        faiss.write_index(query_index, "query.index")
    elif add_delete_update == 1:
        flat_index.remove_ids(np.array([int(event_id)]).astype('int64'))
        faiss.write_index(flat_index, "online_flat.index")
    elif add_delete_update == 2:
        dim = 1024
        k = 1000
        item_vectors = np.array([query_index.reconstruct(i) for i in range(query_index.ntotal)])
        quantiser = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, k, faiss.METRIC_L2)
        index.train(item_vectors)

        flat_df = pd.read_csv('evants_df.csv')
        active_db = flat_df[flat_df['active'] == 1]
        indices = active_db.index.values
        # print(indices)
        active_vectors = np.array([query_index.reconstruct(int(i)) for i in indices])
        index.add_with_ids(active_vectors, indices)
        index.nprobe = 16
        faiss.write_index(index, "online_flat.index")
    return 1
