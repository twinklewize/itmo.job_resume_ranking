from gensim.models.doc2vec import Doc2Vec
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import torch
from config import Config

config = Config()

__doc2vec_model = Doc2Vec.load(config.REPOSITORY_PATH + config.MODELS_PATH + config.DOC2VEC_MODEL)
__bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME)
__tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
__doc2vec_model.random.seed(config.SEED)

def __preprocess_text(text):
    text = text.lower()
    pattern = re.compile("[^а-яА-Яa-zA-Z0-9\-.,;]+")
    text = pattern.sub(" ", text)
    text = ' '.join(text.split())
    return text

def __get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        encoded_input = __tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=config.BERT_MAX_LENGTH)
        with torch.no_grad():
            model_output = __bert_model(**encoded_input)
        emb = model_output.last_hidden_state.mean(dim=1)
        if emb.shape[-1] != 768:
            emb = np.zeros((1, 768))
        else:
            emb = emb.detach().cpu().numpy()
            emb = np.squeeze(emb)
            if emb.shape[0] != 768:
                emb = np.zeros(768)
        embeddings.append(emb)
    return np.array(embeddings)

def __combine_embeddings(bert_embeddings, texts, doc2vec_weight=1.0, bert_weight=0.5):
    combined_embeddings = []
    for bert_emb, text in zip(bert_embeddings, texts):
        doc2vec_emb = __doc2vec_model.infer_vector(text.split())
        doc2vec_emb = np.atleast_1d(doc2vec_emb)
        weighted_doc2vec_emb = doc2vec_emb * doc2vec_weight
        weighted_bert_emb = bert_emb * bert_weight
        combined_emb = np.concatenate((weighted_doc2vec_emb, weighted_bert_emb), axis=0)
        combined_embeddings.append(combined_emb)
    return combined_embeddings

def rank_items_doc2vec(desc, items, threshold=0.5):
    desc = __preprocess_text(desc)

    desc_vector = __doc2vec_model.infer_vector(desc.split()).reshape(1, -1)
    ranked_items = []
    for item in items:
        item = __preprocess_text(item)
        item_vector = __doc2vec_model.infer_vector(item.split()).reshape(1, -1)
        similarity = cosine_similarity(desc_vector, item_vector)[0][0]
        if similarity >= threshold:
            ranked_items.append((item, similarity))
    ranked_items.sort(key=lambda x: x[1], reverse=True)
    return [(item, round(similarity * 100, 2)) for item, similarity in ranked_items]

def rank_items_bert(desc, items, threshold=0.5):
    desc = __preprocess_text(desc)
    desc_bert_emb = __get_bert_embeddings([desc])[0]
    items_bert_embs = __get_bert_embeddings(items)
    similarities = cosine_similarity([desc_bert_emb], items_bert_embs)[0]
    ranked_items = [(item, similarity) for item, similarity in zip(items, similarities) if similarity >= threshold]
    ranked_items.sort(key=lambda x: x[1], reverse=True)
    return [(item, round(similarity * 100, 2)) for item, similarity in ranked_items]

def rank_items_doc2vec_bert(desc, items, threshold=0.5):
    desc = __preprocess_text(desc)
    desc_bert_embs = __get_bert_embeddings([desc])
    desc_vector = __combine_embeddings(desc_bert_embs, [desc])[0]
    items_bert_embs = __get_bert_embeddings(items)
    combined_new_resume_embs = __combine_embeddings(items_bert_embs, items)
    similarities = cosine_similarity([desc_vector], combined_new_resume_embs)[0]
    ranked_items = [(item, similarity) for item, similarity in zip(items, similarities) if similarity >= threshold]
    ranked_items.sort(key=lambda x: x[1], reverse=True)
    return [(item, round(similarity * 100, 2)) for item, similarity in ranked_items]