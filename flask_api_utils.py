from typing import Dict, List
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import torch
import faiss
import json
from langdetect import detect
import os
import nltk
from nltk.corpus import stopwords
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.model_utils import KNRM

GLUE_QQP_DIR = "./data/QQP"

def load_vocab(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def read_glove_embeddings(file_path: str) -> Dict[str, np.array]:
    # допишите ваш код здесь 
    d = {}
    with open(file_path, "r") as a_file:
        for line in a_file:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            d[word] = wordEmbedding
            
    return d

def get_index() -> Dict[str, str]:
    train = pd.read_csv(f'{GLUE_QQP_DIR}/train.tsv', sep='\t')
    index = {
        **{str(train.loc[i, 'qid1']) : train.loc[i, 'question1'] for i in range(train.shape[0])},
        **{str(train.loc[i, 'qid2']) : train.loc[i, 'question2'] for i in range(train.shape[0])}
    }
    return index

class QuestionGrader:
    def __init__(
        self,
        knn_rec_size: int,
        rec_size: int
    ):
        with open('model_config.json') as f:
            model_config = json.load(f)
            
        self.state_intialized = False
        self.knrm_kernel_num = model_config['knrm_kernel_num']
        self.knrm_out_mlp = model_config['knrm_out_mlp']
        self.emb_size = model_config['emb_size']
        self.index_initialized = False
        self.knn_rec_size = knn_rec_size
        self.rec_size = rec_size
        self.tokenizer = nltk.RegexpTokenizer("\w+")
        self.stopwords = set(stopwords.words('english'))

    def init_state(
        self,
        emb_path_knrm: str,
        mlp_path: str,
        vocab_path: str,
        emb_path_glove:str,
        index: Dict[str, str]
    ):
        # загружаем эмбеддинги
        embs = torch.load(emb_path_knrm)
        self.model = KNRM(
            embs['weight'],
            freeze_embeddings=True,
            out_layers=self.knrm_out_mlp,
            kernel_num=self.knrm_kernel_num
        )
        # загружаем модель
        model_state = torch.load(mlp_path)
        model_state['embeddings.weight'] = embs['weight']
        self.model.load_state_dict(model_state)
        self.model.eval()
        # загружаем словарь токенов
        self.vocab = load_vocab(vocab_path)
        self.embs_glove = read_glove_embeddings(emb_path_glove)
        self._create_index(index)
        self.state_intialized = True
        pass

    def _tokenize(self, text: str, remove_stop_words: bool=True):
        tokens = self.tokenizer.tokenize(text.lower())
        if not remove_stop_words:
            return tokens
        else:
            return [token for token in tokens if token not in self.stopwords]

    # creating emb for text search
    def _search_emb(self, text: str):
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        tf_idfs = {i : counts[i]/len(tokens) * self.idf[i] for i in counts}
        sum_tf_idf = sum(tf_idfs.values())
        
        emb = np.zeros(self.emb_size, dtype=np.float32)
        denum = 0

        for token in tokens:
            if (token in self.embs_glove) and (token not in self.stopwords):
                emb += self.embs_glove[token] * (tf_idfs[token]/sum_tf_idf)

        return emb

    # tokenization for knrm model
    def _knrm_tokenize(self, texts: List[str]):
        tokenized = []
        max_len = 0

        for text in texts:
            tokens = self._tokenize(text)
            max_len = max(max_len, len(tokens))
            tokenized.append([self.vocab[token] if token in self.vocab else self.vocab['OOV'] for token in tokens])

        # res: (len(texts), max_len)
        res = torch.LongTensor([i + (max_len-len(i))*[self.vocab['PAD']] for i in tokenized])

        return res

    def _rank_mlp(self, funnel1: Dict[str, np.array]):
        """
        ranking for each query documents from funnel1
        """
        queries, doc_ids = list(funnel1.keys()), list(funnel1.values())
        funnel2 = {}

        for i in range(len(queries)):
            ## creating batch for each query:
            batch = {
                'query' : self._knrm_tokenize(len(doc_ids[i]) * [queries[i]]),
                'document' : self._knrm_tokenize(self.text[doc_ids[i]]),
            }
            with torch.no_grad():
                logits = self.model.predict(batch).flatten().numpy()
            funnel2[queries[i]] = doc_ids[i][logits.argsort()[::-1][:self.rec_size]]

        return funnel2
    
    def _build_idf(self, corpus: List[str]):
        def preprocess(txt: str, tokenizer, stops):
            return ' '.join([i for i in tokenizer.tokenize(txt.lower()) if i not in stops])
        
        preprocess = partial(preprocess, tokenizer=self.tokenizer, stops=self.stopwords)
        vectorizer = TfidfVectorizer(preprocessor=preprocess)
        vectorizer.fit(corpus)
        self.idf = defaultdict(lambda : np.log(len(corpus)), dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)))
        pass

    def _create_index(self, docs: Dict[str, str]):
        text, ids = [], []
        for id in docs:
            ids.append(id)
            text.append(docs[id])

        self._build_idf(text)
        self.ids, self.text = np.array(ids), np.array(text)
        dat = np.array([self._search_emb(s) for s in self.text], dtype=np.float32)
        self.index = faiss.index_factory(self.emb_size, 'Flat')
        self.index.train(dat)
        self.index.add(dat)
        self.index_initialized = True
        
        pass

    def process_query(self, queries: List[str]):
        lang_check = [detect(query) == 'en' for query in queries]

        if sum(lang_check) == 0:
            return lang_check, len(queries)*[None]

        search_embs = np.array([self._search_emb(queries[i]) for i in range(len(queries)) if lang_check[i]], dtype=np.float32)
        # first funnel: knn
        D, I = self.index.search(search_embs, min(self.knn_rec_size, self.index.ntotal))
        
        j = 0
        funnel1 = {}
        for i in range(len(queries)):
            if lang_check[i]:
                funnel1[queries[i]] = I[j]
                j += 1

        funnel2 = self._rank_mlp(funnel1)
        suggestions = []
        for i in range(len(queries)):
            if lang_check[i]:
                text_indices = funnel2[queries[i]]
                suggestions.append([(self.ids[i], self.text[i]) for i in text_indices])
            else:
                suggestions.append(None)
        
        return lang_check, suggestions