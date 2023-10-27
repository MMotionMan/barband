import ms2, ms4, ms1
import pandas as pd
import numpy as np
from scipy.stats import entropy
from barband_ml import transformer_model
import faiss


def test_warm_resim_alg(visited_events_ids):
    df = pd.read_csv('events_DB.csv')
    print("Visited events")
    for i, r in enumerate(visited_events_ids[::-1]):
        print(f"{i}) {df.iloc[r].title}")
    print()
    print("=" * 400)
    print()
    relevat_ranks = ms2.concierge_service(visited_events_ids)
    print("Recommendations")
    for i, r in enumerate(relevat_ranks):
        print(f"{i}) {df.iloc[r].title}")
    print()
    print("=" * 400)
    print()
    x_vals_uniq = np.unique(relevat_ranks)
    n = len(relevat_ranks)
    n_uniq = len(x_vals_uniq)
    d = {}
    for x in relevat_ranks:
        if x not in d:
            d[x] = 1
        else:
            d[x] += 1
    prob_list = []
    for k, v in d.items():
        prob_list.append(d[k] / n)
    print(f"Entropy = {round(entropy(prob_list, n_uniq), 5)}")
    print(f"Percent of unique recommendations {round(len(x_vals_uniq) / n, 3)} ")


def test_duplicates(visited_events_ids):
    relevat_ranks = ms2.concierge_service(visited_events_ids)
    x_vals_uniq = np.unique(relevat_ranks)
    n = len(relevat_ranks)
    n_uniq = len(x_vals_uniq)
    d = {}
    for x in relevat_ranks:
        if x not in d:
            d[x] = 1
        else:
            d[x] += 1
    prob_list = []
    for k, v in d.items():
        prob_list.append(d[k] / n)
    print(f"Entropy = {round(entropy(prob_list, n_uniq), 5)}")
    print(f"Percent of unique recommendations {round(len(x_vals_uniq) / n, 3)} ")


def test_cold_resim_alg(user_vector):
    relevat_ranks = ms4.cold_user_recommendation(user_vector)
    df = pd.read_csv('events_DB.csv')
    for i, r in enumerate(relevat_ranks):
        print(f"{i}) {df.iloc[r].title}")


def test_input_output_data():
    category = "исскуство"
    text = '11-я Международная ярмарка современного искусства'
    description = """Одно из самых ожидаемых событий осеннего арт-сезона — ярмарка современного искусства Cosmoscow, которая представит ведущие галереи, а также новые проекты ключевых российских художников. Новой площадкой проведения станет Центральный выставочный комплекс «Экспоцентр», а именно павильон «Форум», который вместит более 70 галерейных стендов.
    Все произведения на стендах галерей доступны для покупки!
    Предпоказ ярмарки для коллекционеров и арт-профессионалов состоится 28 сентября, гостем которого можете стать Вы!"""
    tags = """Ярмарка, Выставка, Современное искусство, 18+"""

    assert type(category) == str
    assert type(text) == str
    assert type(description) == str
    assert type(tags) == str
    print("Input data transformer_model test. Ok")
    text_tensor = transformer_model.text_to_tensor(category, text, description, tags, tokenizer, model,
                                                   embeddings_type='text')

    assert type(text_tensor) == np.ndarray
    assert text_tensor.dtype == np.float32
    assert len(text_tensor) == 1024
    print("Output data transformer_model test. Ok")

    assert type(text_tensor) == np.ndarray
    assert text_tensor.dtype == np.float32
    print("Input data cold user algorithm test. Ok")

    relevat_ranks = ms4.cold_user_recommendation(text_tensor)
    assert type(relevat_ranks) == list
    assert np.array(relevat_ranks).dtype == np.int64
    assert len(relevat_ranks) == 10
    print("Output data cold user algorithm test. Ok")

    visited_events = [17, 1, 4, 61, 15, 63]
    assert type(visited_events) == list
    assert np.array(visited_events).dtype == np.int64
    print("Input data warm user algorithm test. Ok")

    relevat_ranks = ms2.concierge_service(visited_events)
    assert type(relevat_ranks) == np.ndarray
    assert np.array(relevat_ranks).dtype == np.int64
    assert len(relevat_ranks) == 30
    print("Output data warm user algorithm test. Ok")

    flat_index = faiss.read_index("online_flat.index")
    query_index = faiss.read_index("query.index")
    flat_index_ntotal = flat_index.ntotal
    query_index_ntotal = query_index.ntotal
    event_id = 175
    vector = np.random.rand(1024, )
    add_delete_update = 0
    assert type(event_id) == int
    assert type(relevat_ranks) == np.ndarray
    assert type(add_delete_update) == int
    print("Input data in add elements in the index test. Ok")
    ms1.add_update_delete_index(event_id=event_id, vector=vector, add_delete_update=add_delete_update)
    flat_index = faiss.read_index("online_flat.index")
    query_index = faiss.read_index("query.index")
    assert flat_index_ntotal + 1 == flat_index.ntotal
    assert query_index_ntotal + 1 == query_index.ntotal
    print("Output data in add elements in the index test. Ok")
    flat_index_ntotal = flat_index.ntotal
    query_index_ntotal = query_index.ntotal

    event_id = 79
    add_delete_update = 1
    assert type(event_id) == int
    assert type(add_delete_update) == int
    print("Input data in subtract items from the index test. Ok")
    ms1.add_update_delete_index(event_id=event_id, add_delete_update=add_delete_update)
    flat_index = faiss.read_index("online_flat.index")
    query_index = faiss.read_index("query.index")
    assert flat_index_ntotal - 1 == flat_index.ntotal
    assert query_index_ntotal == query_index.ntotal
    print("Output data in subtract items from the index test. Ok")
