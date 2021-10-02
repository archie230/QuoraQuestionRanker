import numpy as np
import string
import torch

import nltk
nltk.download('punkt')

import pandas as pd
from typing import Dict, List, Tuple, Union, Callable
from collections import Counter
from tqdm import tqdm

from utils.metric_utils import ndcg
from utils.model_utils import KNRM

class RankingDataset(torch.utils.data.Dataset):
    def __init__(
        self, index_pairs_or_triplets: List[List[Union[str, float]]],
        idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
        preproc_func: Callable, max_len: int = 30
    ):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        text = self.idx_to_text_mapping[str(idx)]
        tokenized_text = self.preproc_func(text)
        return self._tokenized_text_to_index(tokenized_text)

    def __getitem__(self, idx: int):
        pass

class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        query_id, doc_id1, doc_id2, target = self.index_pairs_or_triplets[idx]
        X1 = {
            'query' : self._convert_text_idx_to_token_idxs(int(query_id)),
            'document' : self._convert_text_idx_to_token_idxs(int(doc_id1))
        }
        X2 = {
            'query' : self._convert_text_idx_to_token_idxs(int(query_id)),
            'document' : self._convert_text_idx_to_token_idxs(int(doc_id2))
        }
        
        return (X1, X2, target)

class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        query_id, doc_id, value = self.index_pairs_or_triplets[idx]
        X = {
            'query' : self._convert_text_idx_to_token_idxs(int(query_id)),
            'document' : self._convert_text_idx_to_token_idxs(int(doc_id))
        }
        target = value
        return X, target

def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels

class Trainer:
    def __init__(
        self, glue_qqp_dir: str, glove_vectors_path: str,
        min_token_occurancies: int = 1,
        random_seed: int = 0,
        emb_rand_uni_bound: float = 0.2,
        freeze_knrm_embeddings: bool = True,
        knrm_kernel_num: int = 21,
        knrm_out_mlp: List[int] = [],
        dataloader_bs: int = 1024,
        train_lr: float = 0.02,
        change_train_loader_ep: int = 10
    ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path

        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies
        )
        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(self.glue_dev_df)
        self.val_dataset = ValPairsDataset(
            self.dev_pairs_for_ndcg, 
            self.idx_to_text_mapping_dev, 
            vocab=self.vocab, oov_val=self.vocab['OOV'], 
            preproc_func=self.simple_preproc
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0, 
            collate_fn=collate_fn, shuffle=False
        )

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object
        )
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        return inp_str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

    def simple_preproc(self, inp_str: str) -> List[str]:
        inp_str = self.hadle_punctuation(inp_str)
        inp_str = inp_str.lower()
        return nltk.word_tokenize(inp_str)
    
    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        tokens = list(vocab.keys())
        for token in tokens:
            if vocab[token] < min_occurancies:
                vocab.pop(token)
        return vocab
    
    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        text_corpus = []
        for df in list_of_df:
            text_corpus += [df.text_left, df.text_right]
        text_corpus = pd.concat(text_corpus).unique()
        text_tokenized = []
        for s in text_corpus:
            text_tokenized += self.simple_preproc(s)
        vocab = Counter(text_tokenized)
        vocab = self._filter_rare_words(vocab, min_occurancies)
        
        return list(vocab.keys())

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, np.array]:
        d = {}
        with open(file_path, "r") as a_file:
            for line in a_file:
                splitLines = line.split()
                word = splitLines[0]
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                d[word] = wordEmbedding
                
        return d

    def create_glove_emb_from_file(
        self, file_path: str, inner_keys: List[str],
        random_seed: int, rand_uni_bound: float
    ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        np.random.seed(random_seed)
        
        # creating vocab and unk_words list
        glove_embs = self._read_glove_embeddings(file_path)
        embs_len = len(list(glove_embs.values())[0])
        vocab = {
            'PAD' : 0,
            'OOV' : 1
        }
        unk_words = ['PAD', 'OOV']
        for i, j in enumerate(inner_keys):
            vocab[j] = i+2
            if j not in glove_embs:
                unk_words.append(j)
        
        # creating embs matrix
        def get_emb(word: str):
            if word == 'PAD':
                return np.zeros(embs_len)
            elif (word == 'OOV') or (word in unk_words):
                return np.random.uniform(
                    low=-rand_uni_bound, high=rand_uni_bound, size=embs_len
                )
            else:
                return glove_embs[word]

        emb_matrix = np.array([
            get_emb(i)
            for i in vocab
        ])
        
        return emb_matrix, vocab, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(
        self,
        inp_df: pd.DataFrame,
        seed: int,
        min_group_size: int = 2,
        sample_size: int = 10240
    ) -> List[List[Union[str, float]]]:

        np.random.seed(seed)
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        # remove requests that have few marked up documents
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_train_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index
        )
        # sample sample_size leftid with weight = sum (label)
        leftid_weights = inp_df_select[
            inp_df_select.isin(glue_train_leftids_to_use)
        ].groupby('id_left').agg({'label' : 'sum'}).reset_index()
        # конверт веса в вероятности
        ps = np.exp(leftid_weights.label.values)/sum(np.exp(leftid_weights.label.values))
        glue_train_leftids_to_use = np.random.choice(
            leftid_weights.id_left.values, sample_size, replace=False, p=ps
        )
        # filter by left id
        inp_df_select = inp_df_select[
            inp_df_select.id_left.isin(glue_train_leftids_to_use)
        ].reset_index(drop=True)

        # sampling sample_size pairs (id_right, id_right)
        out = inp_df_select.merge(inp_df_select, on='id_left')
        out = out[out['id_right_y'] != out['id_right_x']].reset_index(drop=True)
        out['target'] = (out['label_x'] > out['label_y']).astype(int)

        # pair weight depending on the total target and the target of the whole pair
        def pair_score(x) -> int:
            if x[0] == 1:
                return 10000
            elif x[1] == 2:
                return 10
            elif x[1] == 1:
                return 1000
            else:
                return 100

        out['score_sum'] = out['label_x'] + out['label_y']
        out['sample_weight'] = out[['target', 'score_sum']].apply(pair_score, axis=1)
        out = out.sample(
            n=sample_size,
            replace=False,
            weights=out.sample_weight,
            random_state=seed
        ).reset_index()
        out = out[['id_left', 'id_right_x', 'id_right_y', 'target']]

        return out.values.tolist()

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        return ndcg(ys_true, ys_pred, k=ndcg_top_k)

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])
        model.eval()

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds
        
        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def _get_train_dataloader(self, seed: int) -> torch.utils.data.DataLoader:
        train_dataset = TrainTripletsDataset(
            self.sample_data_for_train_iter(self.glue_train_df, seed), 
            self.idx_to_text_mapping_train,
            vocab=self.vocab,
            oov_val=self.vocab['OOV'],
            preproc_func=self.simple_preproc
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.dataloader_bs,
            num_workers=0, 
            collate_fn=collate_fn,
            shuffle=True
        )
        return train_dataloader

    def train(self, n_epochs: int):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()        
        train_dataloader = self._get_train_dataloader(0)
        ndcgs = []

        epoch_states = {}
        
        for epoch in tqdm(range(n_epochs)):
            # refresh dataloader each change_train_loader_ep
            if (epoch+1) % self.change_train_loader_ep == 0:
                train_dataloader = self._get_train_dataloader(epoch+1)

            self.model.train()
            step = 1
            
            for X1, X2, y_batch in train_dataloader:
                opt.zero_grad()

                batch_pred = self.model.forward(X1, X2)
                loss = criterion(batch_pred, y_batch)
                loss.backward()
                print(f'[TRAIN] epoch: {step} \t loss: {loss.item()}')

                opt.step()
                step += 1

            with torch.no_grad():
                ndcgs.append(self.valid(self.model, self.val_dataloader))

            print(f'[VALIDATION] epoch: {epoch+1} \t ndcg: {ndcgs[-1]}')
            epoch_states[epoch] = [self.model.state_dict(), ndcgs[-1]]
        
        return epoch_states