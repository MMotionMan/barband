from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity


def text_to_tensor(category, text, description, tags, tokenizer, model, embeddings_type='text',
                   concat_type='cat'):
    for item in tags:
        tags = ' '.join(item)
    marked_text = category + '[sep]' + text + '[sep]' + description + '[sep]' + tags

    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

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

        return sentence_embedding
#
#
# with open("input_data.json") as f:
#     data = json.load(f)
#
# with open("input_data_dima.json") as f:
#     data = data + json.load(f)
#
# tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
# model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru", output_hidden_states=True)
#
# tensors = []
# for text in tqdm(data):
#     text_tensor = text_to_tensor(text['category'], text['title'], text['description'][:511], text['tags'], tokenizer, model)
#     tensors.append(text_tensor)
#
# category_data = {}
#
# for i in tqdm(range(len(data))):
#     if data[i]['category'] in category_data:
#         category_data[data[i]['category']].append(tensors[i])
#     else:
#         category_data[data[i]['category']] = [tensors[i]]
#
# cat = list(category_data.keys())
# dist_other_category = []
#
# for i in tqdm(range(1, len(cat))):
#     sum_dist = 0
#     c_dist = 0
#     for item1, item2 in zip(category_data[cat[i]], category_data[cat[i - 1]]):
#         sum_dist += torch.cdist(item1.reshape(1, -1), item2.reshape(1, -1))[0][0]
#         c_dist += 1
#     dist_other_category.append(sum_dist / c_dist)
#
#
# dist_one_category = []
# for i in tqdm(range(len(cat))):
#     sum_dist = 0
#     c_dist = 0
#     for j in range(1, len(category_data[cat[i]])):
#         sum_dist = torch.cdist(category_data[cat[i]][j].reshape(1, -1), category_data[cat[i]][j - 1].reshape(1, -1))[0][0]
#         c_dist += 1
#     dist_one_category.append(sum_dist / c_dist)
# print(dist_other_category)
# print(dist_one_category)

# for i in tqdm(range(1, len(tensors))):
#     print(data[i]['title'], data[i - 1]['title'])
#     print("dist =", cosine_similarity(tensors[i].reshape(1, -1), tensors[i - 1].reshape(1, -1)))
