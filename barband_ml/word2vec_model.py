from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
import torch


class Word2Vec:
    def __init__(self):
        self.russian_model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru", output_hidden_states=True)
        self.russian_tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self.belarusian_model = AutoModelForTokenClassification.from_pretrained(
            "KoichiYasuoka/bert-base-slavic-cyrillic-upos",
            output_hidden_states=True)
        self.belarusian_tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-slavic-cyrillic-upos")

    def text_to_tensor(self, event_info, language, embeddings_type='word', concat_type='cat'):
        # TODO: Необходимо научиться разлечать уровень категории, чтобы выделить древовидную структуру категорий
        if language == 'Russian':
            model = self.russian_model
            tokenizer = self.russian_tokenizer

        elif language == 'Belarusian':
            model = self.belarusian_model
            tokenizer = self.belarusian_tokenizer

        else:
            return 0

        tags = ''
        categories = ''

        for tag in event_info.tags:
            tags += tag + ' '

        for category in event_info.categories:
            categories += category + ' '

        marked_text = "[sep]" + event_info.title + "[sep]" + tags + "[sep]" \
                      + categories + "[sep]" + event_info.description + "[sep]"
        tokenized_text = tokenizer.tokenize(marked_text)
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
                token_vecs_cat = []

                for token in token_embeddings:
                    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

                    token_vecs_cat.append(cat_vec)

                return token_vecs_cat

            elif concat_type == 'sum':
                token_vecs_sum = []

                for token in token_embeddings:
                    sum_vec = torch.sum(token[-4:], dim=0)

                    token_vecs_sum.append(sum_vec)

                return token_vecs_sum

        elif embeddings_type == 'text':

            """ Sentence embeddings """
            token_vecs = hidden_states[-2][0]

            sentence_embedding = torch.mean(token_vecs, dim=0)
            sentence_embedding = sentence_embedding.numpy()
            return list(sentence_embedding)
