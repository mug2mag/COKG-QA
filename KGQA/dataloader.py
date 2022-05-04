import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
# from transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer


class DatasetCOKG(Dataset):
    def __init__(self, data, head_tails_map, entities, entity2idx, type2id=None, type_embeddings=None):
        self.data = data
        self.entities = entities
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        # self.tokenizer_class = RobertaTokenizer
        self.tokenizer_class = BertTokenizer
        # self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained('../pretrained/chinese-roberta-wwm-ext',
                                                              local_files_only=True)
        self.head_tails_map = head_tails_map
        # self.pretrained_weights = 'hfl/chinese-roberta-wwm-ext'
        # self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

        self.type2id = type2id
        self.type_embeddings = type_embeddings

    def __len__(self):
        return len(self.data)

    def pad_sequence(self, arr, max_len=128):
        if len(arr) > max_len:
            print("arr", arr)
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices, gt_type='entity'):
        if type(indices) != list:
            indices = [indices]
        indices = torch.LongTensor(indices)
        # batch_size = len(indices)
        if gt_type == 'entity':
            vec_len = len(self.entity2idx)
        elif gt_type == 'type':
            vec_len = len(self.type2id)
        else:
            vec_len = None
            exit(1)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def pad_zero(self, list_to_pad):
        max_lenth = len(self.entity2idx)
        for i in range(max_lenth):
            if i > len(list_to_pad) - 1:
                list_to_pad.append(0)
        # print('len list_to_pad:', len(list_to_pad))
        return list_to_pad

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id_ = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            # TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
            else:
                print('tail: {} not in entity2id.txt'.format(tail_name))
        tail_onehot = self.toOneHot(tail_ids)

        head_type, tail_type = data_point[3], data_point[4]
        head_id, tail_id = self.type2id[head_type], self.type2id[tail_type]
        head_embedding, tail_embedding = self.type_embeddings[head_id], self.type_embeddings[tail_id]
        tail_type_onehot = self.toOneHot(tail_id, 'type')

        return question_tokenized, attention_mask, head_id_, tail_onehot, data_point[1], \
            head_embedding, tail_embedding, tail_id, tail_type_onehot

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)


# def _collate_fn(batch):
#     print(len(batch))
#     exit(0)
#     question_tokenized = batch[0]
#     attention_mask = batch[1]
#     head_id = batch[2]
#     tail_onehot = batch[3]
#     question_tokenized = torch.stack(question_tokenized, dim=0)
#     attention_mask = torch.stack(attention_mask, dim=0)
#     return question_tokenized, attention_mask, head_id, tail_onehot 

class DataLoaderCOKG(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderCOKG, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
