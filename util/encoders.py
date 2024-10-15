#!/usr/bin/env python3

class BertST:
    ## from sentence_transformers import SentenceTransformer
    ## model = SentenceTransformer('bert-base-uncased')
    ## model.save('data/models/bert-base-uncased')
    ## model = SentenceTransformer('data/models/bert-base-uncased')
    from sentence_transformers import SentenceTransformer
    def __init__(self, bert_path="data/models/bert-base-uncased"):
        self.model = self.SentenceTransformer(bert_path)

    def embed_batch(self, sents, batch_size):
        return self.model.encode(sents, batch_size=batch_size)

    def __call__(self, sents, batch_size=256):
        res = self.embed_batch(sents, batch_size)
        # print(f"res={{len={len(res)}, data=[[{res[0][0]}, {res[0][1]}, ...],")
        # if len(sents)>1:
        #     print(f"                  [{res[1][0]}, {res[1][1]}, ...], ...]}}")
        return res

class bert_encoder:

    def __init__(self):
        self.bert = BertST()
        self.cache = dict()

    @property
    def name(self):
        return "BERT"

    @property
    def embedding_size(self):
        return 768

    def encode(self, text, cache=False):
        res = self.cache.get(text, None)
        if res is not None: return res

        res = self.bert.embed_batch([text], 1)[0]
        if cache: self.cache[text] = res
        return res

    def encode_batch(self, texts, batch_size=1000):
        return self.bert.embed_batch(texts, batch_size)

    def to_numpy(self, v):
        import numpy
        return v.astype(numpy.float32)

class use_encoder:
    import tensorflow_hub
    def __init__(self):
        # self.use = tensorflow_hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
        self.use = self.tensorflow_hub.load('data/universal-sentence-encoder-large_5')
        self.cache = dict()

    @property
    def name(self):
        return "USE"

    @property
    def embedding_size(self):
        return 512

    def encode(self, text, cache=False):
        res = self.cache.get(text, None)
        if res is not None: return res

        res = self.use([text])[0]
        if cache: self.cache[text] = res
        return res

    def encode_batch(self, texts):
        return self.use(texts)

    def to_numpy(self, v):
        return v.numpy()

def get_encoder(encoder_name):
    encoder_name = encoder_name.lower()
    if encoder_name == 'bert':
        return bert_encoder()
    elif encoder_name == 'ts' or encoder_name == 'use':
        return use_encoder()
    else:
        return None

if __name__ == '__main__':
    print('NOTE: encoders.py is deprecated, please use nlu.py as standalone')
    print('NOTE: executable instead.')

    # DEPRECATED: remove the following code and move to nlu.py
    import sys
    import util.levenshtein as levenshtein
    from sklearn.metrics.pairwise import cosine_similarity

    def get_similarity(encoder, text1, text2):
        e1 = encoder.encode(text1)
        e2 = encoder.encode(text2)
        return cosine_similarity([e1], [e2])[0]

    text1 = sys.argv[1]
    text2 = sys.argv[2]
    # print('loading BERT...')
    # bert = bert_encoder()
    print('loading USE...')
    enc_USE = use_encoder()

    # print(f'cos_bert({text1}, {text2}) = {get_similarity(bert, text1, text2)}')
    print(f'cos_use({text1}, {text2}) = {get_similarity(enc_USE, text1, text2)}')

    def sim(s1, s2):
        print(f'cos_use({s1}, {s2}) = {get_similarity(enc_USE, s1, s2)}')
        print(f'lev_sim({s1}, {s2}) = {levenshtein.similarity(s1, s2)}')

    print('sim(s1, s2) returns the different similarities')
    import code
    code.interact(local=locals())
