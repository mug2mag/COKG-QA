import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
# from transformers.modeling_bert import BertModel
from transformers import BertModel


class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, idx2entity, relation_dim, pretrained_embeddings, device,
                 entdrop=0.0, reldrop=0.0, scoredrop=0.0, l3_reg=0.0, model='ComplEx', ls=0.0, do_batch_norm=True,
                 freeze=True, entity2idx=None,
                 train_head_tails_map=None, eval_head_tails_map=None,
                 batch_size=None, relation_embedding=None, type_embedding=None,
                 relation2entity=None, args=None, ent_id2type=None, mat_type2ent=None):
        super(RelationExtractor, self).__init__()

        self.idx2entity = idx2entity
        self.ent_id2type = ent_id2type
        self.mat_type2ent = mat_type2ent
        self.pretrained_embeddings = pretrained_embeddings
        pretrained_embeddings = tuple(torch.Tensor(embed) for embed in pretrained_embeddings)
        self.device = device
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.entity2idx = entity2idx
        self.do_batch_norm = do_batch_norm
        self.train_head_tails_map = train_head_tails_map
        self.eval_head_tails_map = eval_head_tails_map
        self.batch_size = batch_size
        if not self.do_batch_norm:
            print('Not doing batch norm')
        # self.roberta_pretrained_weights = 'roberta-base'
        # self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        self.roberta_model = BertModel.from_pretrained('../pretrained/chinese-roberta-wwm-ext')
        # self.roberta_model = BertModel.from_pretrained('/home/duhuifang/git_local/EmbedKGQA/pretrained_models/albert_zh/prev_trained_model/albert_tiny_zh')
        # self.roberta_model = RobertaModel.from_pretrained('../../RoBERTa_zh_L12/')
        for param in self.roberta_model.parameters():
            param.requires_grad = True

        multiplier = 2
        self.getScores = self.ComplEx
        print('Model is', self.model)

        self.hidden_dim = 768
        self.relation_dim = relation_dim * multiplier  # 200*2

        # self.num_entities = num_entities
        # self.loss = torch.nn.BCELoss(reduction='sum')
        self.loss = self.kge_loss

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        # self.pretrained_embeddings = pretrained_embeddings
        # random.shuffle(pretrained_embeddings)
        # print(pretrained_embeddings[0])
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings, dim=0), freeze=self.freeze)
        relation_embedding = list(relation_embedding.values())
        self.relation_embedding = nn.Embedding.from_pretrained(torch.stack([torch.tensor(rel_emb)
                                                                            for rel_emb in relation_embedding], dim=0),
                                                               freeze=self.freeze)
        self.type_embedding = nn.Embedding.from_pretrained(torch.stack([torch.tensor(type_emb, dtype=torch.float)
                                                                        for type_emb in type_embedding], dim=0),
                                                           freeze=self.freeze)

        print(len(ent_id2type))
        self.tail_type_embedding = \
            torch.stack([torch.sum(self.type_embedding(torch.from_numpy(np.array(t)).long()), 0)
                         for _, t in ent_id2type.items()]).to(self.device)
        print(self.tail_type_embedding.shape)

        # self.relation2entity = relation2entity
        # self.rel2ent = torch.zeros(len(self.relation_embedding.weight), len(self.embedding.weight)).to(self.device)
        # index = list(relation2entity.keys())
        # ent_index = list(relation2entity.values())
        # index = [ind for i, ind in enumerate(index) for _ in ent_index[i]]
        # ent_index = [e for ent in ent_index for e in ent]
        # self.rel2ent[index, ent_index] = 1

        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        # print(self.embedding.weight.shape)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # self.embedding.weight.requires_grad = False
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512

        # self.lin1 = nn.Linear(self.hidden_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        # self.lin4 = nn.Linear(self.mid3, self.mid4)
        # self.hidden2rel = nn.Linear(self.mid4, self.relation_dim)
        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        # self.hidden2rel_base = nn.Linear(self.mid2, self.relation_dim)

        # self.rel2real = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)
        # self.rel2imag = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)
        # self.combine = nn.Linear(self.relation_dim, self.relation_dim)

        self.project_head = nn.Linear(self.relation_dim, self.relation_dim)
        self.project_type = nn.Linear(self.relation_dim, self.relation_dim)
        self.project_type_2 = nn.Linear(self.relation_dim, self.relation_dim)
        self.project_question_2 = nn.Linear(self.relation_dim, self.relation_dim)

        self.project_tail_1 = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)
        self.project_tail_2 = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)

        self.project_tail_type_1 = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)
        self.project_tail_type_2 = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)

        self.project_tail_re = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)
        self.project_tail_im = nn.Linear(self.relation_dim // 2, self.relation_dim // 2)

        self.bn0 = torch.nn.BatchNorm1d(multiplier)
        self.bn2 = torch.nn.BatchNorm1d(multiplier)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self._klloss = torch.nn.KLDivLoss(reduction='sum')

        with open(os.path.join(args.data_path, 'tail_entity2domain.json'), 'r', encoding='utf-8') as fd:
            self.tail_entity2domain = json.load(fd)

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.lin3(outputs)
        # outputs = F.relu(outputs)
        # outputs = self.lin4(outputs)
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)

        # real, imag = torch.split(outputs, self.relation_dim // 2, 1)
        # real = self.rel2real(real)
        # imag = self.rel2imag(imag)
        # outputs = torch.cat([real, imag], 1)
        # outputs = self.combine(outputs)

        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def ComplEx(self, head, relation, score_type='entity', head_embedding=None):
        batch_size = head.size(0)

        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)
        # print('head.shape after self.bn0(head)', head.shape)
        head = self.ent_dropout(head)
        # print('head.shape after self.ent_dropout(head)', head.shape)

        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        # print('re_head.shape and im_head.shape', re_head.shape, im_head.shape)

        # re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        
        if score_type == 'entity':
            re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)
        elif score_type == 'type':
            re_tail, im_tail = torch.chunk(self.type_embedding.weight, 2, dim=1)
        else:
            re_tail, im_tail = None, None
            exit(1)
        # print('re_tail.shape and im_tail.shape', re_tail.shape, im_tail.shape)
        re_tail = self.project_tail_1(re_tail)
        im_tail = self.project_tail_2(im_tail)

        if score_type == 'entity':
            re_tail_type, im_tail_type = torch.chunk(self.tail_type_embedding, 2, dim=1)
        else:
            re_tail_type, im_tail_type = None, None
            exit(1)
        print("re_tail size: ", re_tail.size())
        print("re_tail_type size: ", re_tail_type.size())
        re_tail_type = self.project_tail_type_1(re_tail_type)
        im_tail_type = self.project_tail_type_2(im_tail_type)

        re_tail = re_tail + re_tail_type
        im_tail = im_tail + im_tail_type
        

        # n = re_tail.size(0)
        #
        # re_tail = re_tail.unsqueeze(0).repeat(batch_size, 1, 1)
        # im_tail = im_tail.unsqueeze(0).repeat(batch_size, 1, 1)

        # re_head_type, im_head_type = torch.split(head_embedding, self.relation_dim // 2, 1)
        # re_head_type = self.project_tail_re(re_head_type)
        # im_head_type = self.project_tail_im(im_head_type)
        # re_head_type = re_head_type.unsqueeze(1).repeat(1, n, 1)
        # im_head_type = im_head_type.unsqueeze(1).repeat(1, n, 1)
        #
        # re_tail = re_tail + re_head_type
        # im_tail = im_tail + im_head_type
        # re = torch.cat([re_tail, re_head_type], 2).reshape(-1, self.relation_dim)
        # im = torch.cat([im_tail, im_head_type], 2).reshape(-1, self.relation_dim)

        # re_tail = self.project_tail_re(re)
        # im_tail = self.project_tail_im(im)

        # re_tail = re_tail.reshape(batch_size, -1, self.relation_dim // 2)
        # im_tail = im_tail.reshape(batch_size, -1, self.relation_dim // 2)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))

        # re_score = re_score.unsqueeze(1)
        # im_score = im_score.unsqueeze(1)
        #
        # score = torch.bmm(re_score, re_tail.permute(0, 2, 1)) + torch.bmm(im_score, im_tail.permute(0, 2, 1))

        # pred = torch.sigmoid(score)
        pred = score
        # print('pred.shape:', pred.shape)

        return pred

    def TransH(self, head, relation):
        head = self.ent_dropout(head)
        t = self.embedding.weight.transpose(1, 0)

        relation = self.rel_dropout(relation)
        norm_relation, hyper_relation = torch.chunk(relation, 2, dim=1)

        score = ((head - torch.mul(torch.mm(norm_relation, head), head)) + hyper_relation - (
                t - torch.mul(torch.mm(norm_relation, t), t))).sqrt()
        return score

    def get_entities_in_domain(self, tail_domains):
        entities_in_domain = []
        for entity, entity_domains in self.tail_entity2domain.items():
            if entity in tail_domains:
                entity_id = self.entity2idx[entity]
                entities_in_domain.append(entity_id)
        return entities_in_domain

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def confine_tail_domain_embedding(self, tail_one_hots):
        tail_entitys = []
        for tail_one_hot in tail_one_hots:
            for id, i in enumerate(tail_one_hot):
                if i != torch.tensor(1):
                    continue
                tail_entity = self.idx2entity[id]
                tail_entitys.append(tail_entity)

        for tail_entity in tail_entitys:
            tail_domains = self.tail_entity2domain.get(tail_entity)
            if tail_domains:
                # print("tail_domian", tail_domains)
                entities_in_domain = self.get_entities_in_domain(tail_domains)
                for entity_id in entities_in_domain:
                    self.pretrained_embeddings[entity_id] = [i * 3 for i in self.pretrained_embeddings]

        pretrained_embeddings = tuple(torch.Tensor(embed) for embed in self.pretrained_embeddings)
        self.embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings, dim=0), freeze=self.freeze).to(
            self.device)
        embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings, dim=0), freeze=self.freeze).to(
            self.device)
        return embedding

    def forward(self, question_tokenized, attention_mask, p_head, p_tail,
                head_embedding, tail_embedding, tail_id, tail_type_gt):
        # single_head = [int(i) for i in list(p_head)]
        # if not self.training:
        #     tails_to_select = [list(self.eval_head_tails_map[s_head]) for s_head in single_head]
        # else:
        #     tails_to_select = [list(self.train_head_tails_map[s_head]) for s_head in single_head]
        #
        # tails_to_select_ = [i for tails in tails_to_select for i in tails]
        #
        # tails_to_type = [[t_ for t in tail for t_ in self.ent_id2type[t]] for tail in tails_to_select]
        #
        # tails_to_tails = \
        #     (torch.from_numpy(np.stack([sum(self.mat_type2ent[types]) for types in tails_to_type])).to(self.device) > 0).float()

        # tails_selector = torch.zeros(len(p_head), len(self.idx2entity)).to(self.device)
        # index = [i for i in range(len(p_head)) for _ in range(len(tails_to_select[i]))]
        # tails_to_select = [i for tails in tails_to_select for i in tails]
        # tails_selector[index, tails_to_select] = 1

        # print("p_head.shape:", p_head.shape)  # torch.Size([64])
        # print("p_tail.shape:", p_tail.shape)  # torch.Size([64, 37186])
        # tails_to_select = []
        # for single_head in p_head:
        #     single_head = int(single_head)
        #     tails_to_select.append(list(self.head_tails_map[single_head]))

        # print("tails_to_select len:", len(tails_to_select))  # torch.Size([64, 37186])
        # tails_selector = [[0]*len(self.entity2idx)]*self.batch_size
        # for single_tails_to_select, single_tails_selector in zip(tails_to_select, tails_selector):
        #     for index in single_tails_to_select:
        #         single_tails_selector[index] = 1

        # print("tails_selector len:", len(tails_selector))  # torch.Size([64, 37186])

        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        # print("question_embedding.shape:", question_embedding.shape)
        rel_embedding = self.applyNonLinear(question_embedding)  # 64x400
        p_head = self.embedding(p_head)

        p_head = self.project_head(p_head)
        p_type = self.project_type(head_embedding.float())

        p_head = p_head + p_type

        # tail_type_embeddings = \
        #     [torch.sum(self.type_embedding(torch.from_numpy(np.array(t)).long().to(self.device)), 0)
        #      for t in tails_to_type]
        # tail_type_embeddings = torch.stack(tail_type_embeddings)

        pred = self.getScores(p_head, rel_embedding, 'entity', p_type)

        # rel_embedding = self.project_question_2(rel_embedding)
        # p_type = self.project_type_2(p_type)
        #
        # type_pred = self.getScores(p_type, rel_embedding, 'type')

        # rel_score = torch.sigmoid(torch.mm(rel_embedding, self.relation_embedding.weight.T)) > 0.5
        # rel_score = rel_score.float()
        # ent_score = torch.mm(rel_score, self.rel2ent)
        #
        # pred = pred + ent_score

        # pred = pred * tails_to_tails

        # if pred.shape[0] != 64:
        #     print("p_head.shape:", p_head.shape)
        #     print("p_head:", p_head)

        # print("self.device ", self.device )
        # last_pred = pred.mul(torch.tensor(tails_selector).to(self.device))
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0 - self.label_smoothing) * actual) + (1.0 / actual.size(1))
        loss = self.loss(pred, actual)
        # loss = self.loss(pred, actual) + self.loss(type_pred, tail_type_gt)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss

    def get_score_ranked(self, head, question_tokenized, attention_mask, confine_flag=False,
                         head_embedding=None, tail_embedding=None):
        # single_head = int(head)
        single_head = [int(i) for i in list(head)]
        tails_to_select = [list(self.eval_head_tails_map[s_head]) for s_head in single_head]

        # tails_to_type = [[t_ for t in tail for t_ in self.ent_id2type[t]] for tail in tails_to_select]
        #
        # tails_to_tails = \
        #     (torch.from_numpy(np.stack([sum(self.mat_type2ent[types]) for types in tails_to_type])).to(self.device) > 0).float()

        # print("tails_to_select len:", len(tails_to_select))  # torch.Size([64, 37186])
        # tails_selector = [0] * len(self.entity2idx)
        # for index in tails_to_select:
        #     tails_selector[index] = 1

        tails_selector = torch.zeros(len(head), len(self.idx2entity)).to(self.device)
        index = [i for i in range(len(head)) for _ in range(len(tails_to_select[i]))]
        tails_to_select = [i for tails in tails_to_select for i in tails]
        tails_selector[index, tails_to_select] = 1

        question_embedding = self.getQuestionEmbedding(question_tokenized, attention_mask)
        rel_embedding = self.applyNonLinear(question_embedding)
        head = self.embedding(head)

        head = self.project_head(head)
        p_type = self.project_type(head_embedding.float())

        head = head + p_type

        scores = self.getScores(head, rel_embedding, 'entity', p_type)

        # rel_score = torch.sigmoid(torch.mm(rel_embedding, self.relation_embedding.weight.T)) > 0.5
        # rel_score = rel_score.float()
        # ent_score = torch.mm(rel_score, self.rel2ent)
        #
        # scores = scores + ent_score
        #
        # scores = scores * tails_to_tails

        if confine_flag:
            # scores = scores.mul(torch.tensor(tails_selector).to(self.device))
            scores = scores.mul(tails_selector)
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        # return last_pred
        return scores
