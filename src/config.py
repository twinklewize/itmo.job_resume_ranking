class Config:
    def __init__(self):
        # Repository path - You should change this to your local repository path.
        self.REPOSITORY_PATH = '/Users/twinklewize/programming/matching'
        self.MODELS_PATH = '/models/'
        self.doc2vec_config()
        self.bert_config()
        self.SEED = 2023

    def doc2vec_config(self):
        self.DOC2VEC_MODEL = 'rjm_doc2vec_30000rows_exp3.model'

    def bert_config(self):
        self.BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
        self.BERT_MAX_LENGTH = 512