from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch


def text_to_tensor(category, text, description, tags, tokenizer, model, embeddings_type='text',
                   concat_type='cat'):
    for item in tags:
        tags = ' '.join(item)
    marked_text = category + '[sep]' + text + '[sep]' + description + '[sep]' + tags

    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    segments_ids = [1] * len(tokenized_text)

    # print(segments_ids)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        hidden_states = outputs['hidden_states']

    # print("Number of layers:", len(hidden_states), "  (initial embeddings + 24 BERT layers)")
    # layer_i = 0
    #
    # print("Number of batches:", len(hidden_states[layer_i]))
    # batch_i = 0
    #
    # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    # token_i = 0
    #
    # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

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
        token_vecs = hidden_states[-1][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding

#
# tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-slavic-cyrillic-upos")
# model = AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/bert-base-slavic-cyrillic-upos",
#                                                         output_hidden_states=True)
#
# text = 'прывітанне, як справы, чым наогул займаешся'
# description = 'прывітанне, як справы, чым наогул займаешся'
# tags = ['прывітанне, як справы, чым наогул займаешся', 'прывітанне, як справы, чым наогул займаешся']
# category = 'прывітанне, як справы, чым наогул займаешся'
#
# print(text_to_tensor(category, text, description, tags, tokenizer, model))
