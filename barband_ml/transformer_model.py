from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import faiss
from scipy.stats import entropy

tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru", output_hidden_states=True)

text_lens = []


def text_to_tensor(category, text, description, tags, tokenizer, model, embeddings_type='word', concat_type='cat'):
    marked_text = category + "[sep]" + text + "[sep]" + tags + "[sep]" + description + "[sep]"
    tokenized_text = tokenizer.tokenize(marked_text)
    text_lens.append(len(tokenized_text))
    if len(tokenized_text) >= 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    # print(segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    if embeddings_type == 'word':
        """Token embeddings"""
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()

        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()

        token_embeddings = token_embeddings.permute(1, 0, 2)

        if concat_type == 'cat':
            # Stores the token vectors, with shape [22 x 3,072]
            token_vecs_cat = []

            for token in token_embeddings:
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

                token_vecs_cat.append(cat_vec)

            # print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

            return token_vecs_cat

        elif concat_type == 'sum':
            # Stores the token vectors, with shape [word_num x 1024]
            token_vecs_sum = []

            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)

                token_vecs_sum.append(sum_vec)

            # print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

            return token_vecs_sum

    elif embeddings_type == 'text':

        """ Sentence embeddings """
        # `token_vecs` is a tensor with shape [num_word x 1024]
        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.numpy()
        return list(sentence_embedding)