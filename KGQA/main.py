import os
import re
from itertools import groupby

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from dataloader import DatasetCOKG, DataLoaderCOKG
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import time
import sys
from configuration_KGQA import config
from torch.utils.data.sampler import \
    (Sampler, SequentialSampler, RandomSampler,
     SubsetRandomSampler, WeightedRandomSampler, BatchSampler)

sys.path.append("../..")  # Adds higher directory to python modules path.
import json
from log_config import Logger
from tensorboardX import SummaryWriter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


embedding_path = config['DEFAULT']['complex_vec']
id2en_path = config['DEFAULT']['id2entity']
id2type_path = config['DEFAULT']['id2type']
QA_data_path = config['DEFAULT']['qa_data']

file_without_confine = config['wrong_answer_analysis']['file_without_confine']
file_with_confined_domain = config['wrong_answer_analysis']['file_with_confined_domain']

parser = argparse.ArgumentParser()

parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--load_from', type=str, default='')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--validate_every', type=int, default=1)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--outfile', type=str, default='best_score_model')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--relation_dim', type=int, default=400)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--freeze', type=str2bool, default=True)
parser.add_argument('--do_batch_norm', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, default='gathered')

parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='../data/COKG_data')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
args = parser.parse_args()
id2entities = None

embedding_path = os.path.join(args.data_path, embedding_path)
# id2en_path = os.path.join(args.data_path, "id2entity.json")

if args.save_path == '':
    args.save_path = time.strftime('%Y-%m-%d-%H-%M')

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
if not os.path.exists(os.path.join('./checkpoints', args.save_path)):
    os.mkdir(os.path.join('./checkpoints', args.save_path))

log = Logger(os.path.join('./checkpoints', args.save_path, 'all.log'))
log.logger.info(str(args))

train_writer = SummaryWriter(os.path.join('./checkpoints', args.save_path, 'train_log'))
test_writer = SummaryWriter(os.path.join('./checkpoints', args.save_path, 'test_log'))


# entity2idx, idx2entity, 嵌入矩阵
def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix


def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word_to_ix:
                idx2word[len(word_to_ix)] = word
                word_to_ix[word] = len(word_to_ix)

        length = len(sent.split())
        if length > maxLength:
            maxLength = length

    return word_to_ix, idx2word, maxLength


def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict, 'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e, r


def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('##')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G


def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key


def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes


def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x


def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result


@torch.no_grad()
def validate_v2(data_path, device, model, dataloader, entity2idx, model_name, writeCandidatesToFile=False,
                confine_flag=False):
    model.eval()
    data = process_text_file(data_path)
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = []
    data_gen = data_generator(data=data, dataloader=dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0



    scores_list = []
    hit_at_10 = 0
    hit_at_3 = 0
    candidates_with_scores = []
    import math
    for _ in tqdm(range(math.ceil(len(data) / args.batch_size))):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]

        head_embedding, tail_embedding = d[5].to(device), d[6].to(device)

        # tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = \
            model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask,
                                   confine_flag=confine_flag,
                                   head_embedding=head_embedding, tail_embedding=tail_embedding)
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(head), len(entity2idx)).to(device)
        mask[list(range(len(head))), head] = 1
        # mask[head] = 1
        # reduce scores of all non-candidates
        new_scores = scores - (mask * 99999)
        pred_ans = torch.argmax(new_scores, 1).detach().cpu().numpy()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)

        # if pred_ans == head.item():
        #     log.logger.info('Head and answer same')
        #     log.logger.info(torch.max(new_scores))
        #     log.logger.info(torch.min(new_scores))

        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
        # print(len(candidates))
        # exit(0)

        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c = torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)

        for n_s, a in zip(new_scores, ans):
            if inTopk(n_s, a, 10):
                hit_at_10 += 1

        for n_s, a in zip(new_scores, ans):
            if inTopk(n_s, a, 3):
                hit_at_3 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0

        q_text = d[4]

        for n_s, a, q, h, p_a in zip(new_scores, ans, q_text, head, pred_ans):
            ans_len = len(a)
            # if pred_ans in ans:
            if inTopk(n_s, a, ans_len):
                total_correct += 1
                is_correct = 1
            else:
                num_incorrect += 1

            answers.append(
                str(h) + '\t' + q + '\t' + str(p_a) + '\t' + '##'.join([str(j) for j in a]) +
                '\t' + str(is_correct))
        # break
        # except:
        #     error_count += 1

    if writeCandidatesToFile:
        # pickle.dump(candidates_with_scores, open('candidates_with_score_and_qe_half.pkl', 'wb'))
        pickle.dump(candidates_with_scores, open('webqsp_scores_full_kg_fixed.pkl', 'wb'))
        log.logger.info('wrote candidate file (for future answer processing)')
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    top10 = hit_at_10 / len(data)
    top3 = hit_at_3 / len(data)
    accuracy = total_correct / len(data)
    # log.logger.info('Error mean rank: %f' % (incorrect_rank_sum/num_incorrect))
    # log.logger.info('%d out of %d incorrect were not in top 50' % (not_in_top_50_count, num_incorrect))
    return answers, accuracy, top10, top3


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    log.logger.info('Wrote to %s' % fname)
    return


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()


def getEntityEmbeddings(embed_path, id2entities_path):
    with open(embed_path, 'r') as load_f:
        ent_embeddings = json.load(load_f)
        ent_embeddings_re, ent_embeddings_im = ent_embeddings['ent_re_embeddings.weight'], ent_embeddings[
            'ent_im_embeddings.weight']
        rel_embeddings_re, rel_embeddings_im = ent_embeddings['rel_re_embeddings.weight'], ent_embeddings[
            'rel_im_embeddings.weight']
    print("ent_embeddings_re len:", len(ent_embeddings_re))
    with open(id2entities_path, 'r', encoding='utf-8') as id2ent:
        global id2entities
        id2entities = json.load(id2ent)
    e = {}
    for i, embedding_re_and_im in enumerate(zip(ent_embeddings_re, ent_embeddings_im)):
        if i >= len(id2entities):
            break
        re_embedding = embedding_re_and_im[0]
        im_embedding = embedding_re_and_im[1]
        re_embedding.extend(im_embedding)
        # print("re_embedding len:", len(re_embedding))
        e[id2entities[format(i)]] = re_embedding

    rel = {}
    for i, embedding in enumerate(zip(rel_embeddings_re, rel_embeddings_im)):
        re_embedding, im_embedding = embedding[0], embedding[1]
        re_embedding.extend(im_embedding)
        rel[id2entities[format(i)]] = re_embedding
    return e, rel


def get_entity2num():
    with open(os.path.join(args.data_path, 'entity2num.json'), 'r', encoding='utf-8') as fr:
        relation2num = json.load(fr)
    return relation2num


def get_rel2entity():
    rel2ent = {}
    with open(os.path.join(args.data_path, 'train2id.txt'), 'r', encoding='utf-8') as f:
        data = f.readlines()[1:]
        data = [d.strip() for d in data]
        for dat in data:
            _, ent, rel = dat.split(' ')
            ent, rel = int(ent), int(rel)
            if rel not in rel2ent:
                rel2ent[rel] = [ent]
            else:
                rel2ent[rel].append(ent)
    return rel2ent


def train(data_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim,
          relation_dim, gpu, use_cuda, patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg,
          model_name, decay, ls, load_from, outfile, do_batch_norm, valid_data_path=None, multi_hops_data=None):
    log.logger.info('Loading entities and relations')

    id2entities_path = id2en_path
    e, rel = getEntityEmbeddings(embedding_path, id2entities_path)

    log.logger.info('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data = process_text_file(data_path, split=False)
    train_head_tails_map = get_validate_head_tails_map(data_path, entity2idx)
    # inferece_head_tails_map = get_validate_head_tails_map(all_data_path, entity2idx)
    inferece_head_tails_map = get_validate_head_tails_map(valid_data_path, entity2idx)

    with open(os.path.join(args.data_path, 'sub_ent2type.json'), 'r', encoding='utf-8') as file:
        ent2type = json.load(file)
    # print("ent2type length：" , len(ent2type))
    # with open(os.path.join(args.data_path, 'type2id.json'), 'r', encoding='utf-8') as file:
    #     type2id = json.load(file)
    # mat_type2ent = np.load(os.path.join(args.data_path, 'mat_type2ent.npy'))
    # ent_id2type = {}
    # for key, value in ent2type.items():
    #     ent_id2type[entity2idx[key]] = [type2id[v] for v in value]

    ent_id2type, mat_type2ent = None, None

    with open(id2type_path, 'r', encoding='utf-8') as file:
        id2type = json.load(file)
    type2id = {}
    for key, value in id2type.items():
        type2id[value] = int(key)
    with open(os.path.join(args.data_path, 'complEx_schema_embedding.vec'), 'r', encoding='utf-8') as file:
        type_embedding = json.load(file)
    type_embedding_re, type_embedding_im = \
        type_embedding['ent_re_embeddings.weight'], type_embedding['ent_im_embeddings.weight']
    type_embedding_re, type_embedding_im = np.array(type_embedding_re), np.array(type_embedding_im)
    type_embeddings = np.stack([type_embedding_re, type_embedding_im], 1).reshape(len(type_embedding_re), -1)

    ent_id2type = {}
    # try:
    for key, value in ent2type.items():
        type_id_list = []
        for v in value:
              if v in type2id:
                type_id_list.append(type2id[v])
        ent_id2type[entity2idx[key]] = type_id_list
            # ent_id2type[entity2idx[key]] = [type2id[v] for v in value]
    # except:
    #     pass
        # print('---')
        # print(key)
        # print('---')

    log.logger.info('Train file processed, making dataloader')
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetCOKG(data, inferece_head_tails_map, e, entity2idx,
                          type2id=type2id, type_embeddings=type_embeddings)
    # entity2num = get_entity2num()
    # weights = [2 if entity2num.get(id2entities[str(item[2])], 0) < 10 else 1 for item in dataset]
    # sampler = WeightedRandomSampler(weights, num_samples=150000, replacement=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True, pin_memory=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True,
                             pin_memory=True)
    log.logger.info('Creating model...')
    # 这里需要的参数：e{name:embedding}，entity2idx, idx2entity, embedding_matrix
    model = RelationExtractor(embedding_dim=embedding_dim, idx2entity=idx2entity, relation_dim=relation_dim,
                              pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop=entdrop,
                              reldrop=reldrop, scoredrop=scoredrop, l3_reg=l3_reg, model=model_name, ls=ls,
                              do_batch_norm=do_batch_norm, entity2idx=entity2idx,
                              train_head_tails_map=train_head_tails_map,
                              eval_head_tails_map=inferece_head_tails_map,
                              batch_size=batch_size, args=args, relation_embedding=rel, type_embedding=type_embeddings,
                              # relation2entity=get_rel2entity(),
                              relation2entity=None,
                              ent_id2type=ent_id2type, mat_type2ent=mat_type2ent)
    log.logger.info('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = load_from
        # fname = "/scratche/home/apoorv/tut_pytorch/kg-qa/checkpoints/roberta_finetune/" + load_from + ".pt"
        model.load_state_dict(torch.load(fname, map_location=torch.device('cuda')))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # def lambda1(e):
    #     if e < 3:
    #         return 0.9 * e / 3 + 0.1
    #     else:
    #         return decay ** (e - 3)

    scheduler = ExponentialLR(optimizer, decay)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

    optimizer.zero_grad()
    best_score = -float("inf")  # inf 是指无限的
    best_score_1 = -float("inf")  # inf 是指无限的
    best_score_2 = -float("inf")  # inf 是指无限的
    best_score_3 = -float("inf")  # inf 是指无限的
    # best_model = model.state_dict()
    one_hop_valid_data, two_hop_valid_data, three_hop_valid_data = multi_hops_data
    no_update = 0
    # time.sleep(10)
    print("data_loader len:", len(data_loader))

    k = 0

    for epoch in range(nb_epochs):  # 每个epoch下面在进行10次train和一次validation?
        # for epoch in range(1):  # 每个epoch下面在进行10次train和一次validation?
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            print("phase:", phase)
            if phase == 'train':
                model.train()
                # model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2].to(device)
                    positive_tail = a[3].to(device)

                    head_embedding, tail_embedding = a[5].to(device), a[6].to(device)

                    tail_id = a[7].to(device)
                    tail_type_gt = a[8].to(device)

                    # tail_to_select = a[4]
                    # negtive_tail = a[4].to(device)
                    # print("positive_head.shape", positive_head.shape)
                    # print("positive_tail.shape", positive_tail.shape)
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask,
                                 p_head=positive_head, p_tail=positive_tail,
                                 head_embedding=head_embedding, tail_embedding=tail_embedding, tail_id=tail_id,
                                 tail_type_gt=tail_type_gt)
                    # - model(question_tokenized=question_tokenized, attention_mask=attention_mask,
                    #       p_head=positive_head, p_tail=negtive_tail)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()

                    k += 1
                    train_writer.add_scalar('loss', loss.item(), global_step=k)

                scheduler.step()

            elif phase == 'valid':
                model.eval()
                eps = 0.0001
                # -------------------------gathered data---------------------------------------------------------
                answers, score, top10, top3 = validate_v2(model=model, data_path=valid_data_path, entity2idx=entity2idx,
                                                    dataloader=dataset, device=device, model_name=model_name,
                                                    confine_flag=False)
                log.logger.info('top10:{}'.format(top10))
                log.logger.info('top3:{}'.format(top3))
                log.logger.info('score:{}'.format(score))

                test_writer.add_scalar('score', score, global_step=epoch)

                if score > best_score + eps:
                    no_update = 0
                    best_model = model.state_dict()

                    get_wrong_results(answers, file_without_confine)

                    # with open('answers_not_freeze.txt', 'w') as f:
                    #     f.write('\n'.join(answers))

                    log.logger.info(
                        "{} hop Validation accuracy (no relation scoring) increased from previous epoch score {}".format(
                            hops, best_score))
                    best_score = score

                    torch.save(best_model,
                               os.path.join('./checkpoints', args.save_path, 'best_score_model.pt'))
                elif (score < best_score + eps) and (no_update < patience):
                    no_update += 1
                    log.logger.info("Validation accuracy decreases to %f from %f, %d more epoch to check" % (
                        score, best_score, patience - no_update))
                elif no_update == patience:
                    log.logger.info("Model has exceed patience. Saving best model and exiting")
                    log.logger.info("--------------------------------------------------------------")

                    # -------------------------gathered data---------------------------------------------------------
                    log.logger.info("Gathered data evaluation:")
                    # all_best_model.eval()
                    model.load_state_dict(torch.load(
                        os.path.join('./checkpoints', args.save_path, 'best_score_model.pt'),
                        map_location=torch.device('cuda')))
                    answers, score, top10, top3 = validate_v2(model=model, data_path=valid_data_path,
                                                              entity2idx=entity2idx,
                                                              dataloader=dataset, device=device,
                                                              model_name=model_name, confine_flag=False)

                    log.logger.info('top10:{}'.format(top10))
                    log.logger.info('top3:{}'.format(top3))
                    log.logger.info('score:{}'.format(score))

                    # ----------------------------------one hop------------------------------------------------------
                    log.logger.info("One_hop data evaluation:")
                    # all_best_model.eval()
                    answers_1, score_1, top10_1, top3_1 = validate_v2(model=model,
                                                                      data_path=one_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name, confine_flag=False)

                    log.logger.info('1-top10:{}'.format(top10_1))
                    log.logger.info('1-top3:{}'.format(top3_1))
                    log.logger.info('1-score:{}'.format(score_1))

                    # ----------------------------------two hop------------------------------------------------------
                    answers_2, score_2, top10_2, top3_2 = validate_v2(model=model,
                                                                      data_path=two_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name,
                                                                      confine_flag=False)

                    log.logger.info('2-top10:{}'.format(top10_2))
                    log.logger.info('2-top3:{}'.format(top3_2))
                    log.logger.info('2-score:{}'.format(score_2))

                    # ----------------------------------three hop------------------------------------------------------
                    answers_3, score_3, top10_3, top3_3 = validate_v2(model=model,
                                                                      data_path=three_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name,
                                                                      confine_flag=False)

                    log.logger.info('3-top10:{}'.format(top10_3))
                    log.logger.info('3-top3:{}'.format(top3_3))
                    log.logger.info('3-score:{}'.format(score_3))

                    # -----------------------------------------------------------------------------------------------
                    # log.logger.info("Model will inferece in the range of the specified head")
                    # answers, score, top10 = validate_v2(model=model, data_path=valid_data_path, entity2idx=entity2idx,
                    #                                     dataloader=dataset, device=device, model_name=model_name,
                    #                                     confine_flag=True)
                    # get_wrong_results(answers, file_with_confined_domain)
                    # # with open('answers_in_confined_domain.txt', 'w') as f:
                    # #     f.write('\n'.join(answers))
                    # log.logger.info("Inferece in the range of the specified head score {}".format(score))
                    # exit()

                    # -------------------------------------------------------------------------------------------------

                    log.logger.info("---------------------------------------------------------------")

                    log.logger.info("Model will inferece in the range of the same head!!!!!!!!!!!!!")
                    # -------------------------gathered data---------------------------------------------------------
                    # all_best_model.eval()
                    answers, score, top10, top3 = validate_v2(model=model, data_path=valid_data_path,
                                                              entity2idx=entity2idx,
                                                              dataloader=dataset, device=device,
                                                              model_name=model_name, confine_flag=True)

                    log.logger.info('head_top10:{}'.format(top10))
                    log.logger.info('head_top3:{}'.format(top3))
                    log.logger.info('head_score:{}'.format(score))
                    log.logger.info(
                        "all data inference in the range of the specified head score {}".format(
                            score))
                    # ----------------------------------one hop------------------------------------------------------
                    # all_best_model.eval()
                    answers_1, score_1, top10_1, top3_1 = validate_v2(model=model,
                                                                      data_path=one_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name, confine_flag=True)

                    log.logger.info('head_1-top10:{}'.format(top10_1))
                    log.logger.info('head_1-top3:{}'.format(top3_1))
                    log.logger.info('head_1-score:{}'.format(score_1))
                    log.logger.info(
                        "1 hop inference in the range of the specified head score {}".format(
                            best_score_1))

                    # ----------------------------------two hop------------------------------------------------------
                    # all_best_model.eval()
                    answers_2, score_2, top10_2, top3_2 = validate_v2(model=model,
                                                                      data_path=two_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name,
                                                                      confine_flag=True)

                    log.logger.info('head_2-top10:{}'.format(top10_2))
                    log.logger.info('head_2-top3:{}'.format(top3_2))
                    log.logger.info('head_2-score:{}'.format(score_2))

                    log.logger.info(
                        "2 hop inference in the range of the specified head score {}".format(
                            best_score_2))

                    # ----------------------------------three hop------------------------------------------------------
                    # all_best_model.eval()
                    answers_3, score_3, top10_3, top3_3 = validate_v2(model=model,
                                                                      data_path=three_hop_valid_data,
                                                                      entity2idx=entity2idx,
                                                                      dataloader=dataset, device=device,
                                                                      model_name=model_name,
                                                                      confine_flag=True)

                    log.logger.info('3-top10:{}'.format(top10_3))
                    log.logger.info('3-top3:{}'.format(top3_3))
                    log.logger.info('3-score:{}'.format(score_3))

                    log.logger.info(
                        "3 hop inference in the range of the specified head score {}".format(
                            best_score_3))
                    # -------------------------------------------------------------------------------------------------
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()

                if epoch == nb_epochs - 1:
                    log.logger.info("Final Epoch has reached. Stoping and saving model.")
                    exit()


def eval(data_path,
         load_from,
         gpu,
         hidden_dim,
         relation_dim,
         embedding_dim,
         hops,
         batch_size,
         num_workers,
         model_name,
         do_batch_norm,
         use_cuda):
    log.logger.info('Loading entities and relations')
    embed_path = embedding_path
    id2entities_path = id2en_path
    e, rel = getEntityEmbeddings(embed_path, id2entities_path)

    log.logger.info('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data, head_tails_map = process_text_file(entity2idx, data_path, split=False)
    log.logger.info('Evaluation file processed, making dataloader')

    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetCOKG(data, e, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log.logger.info('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, idx2entity=idx2entity, relation_dim=relation_dim,
                              pretrained_embeddings=embedding_matrix, device=device,
                              model=model_name, do_batch_norm=do_batch_norm, args=args)
    log.logger.info('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "./checkpoints/roberta_finetune/" + load_from + ".pt"
        log.logger.info('Loading from %s' % fname)
        model.load_state_dict(torch.load(fname, map_location=torch.device('cuda')))
        log.logger.info('Loaded successfully!')
    else:
        log.logger.info('Need to specify load_from argument for evaluation!')
        exit(0)

    model.to(device)
    answers, score, top10 = validate_v2(model=model, data_path=data_path,
                                        entity2idx=entity2idx, dataloader=dataset,
                                        device=device, model_name=model_name,
                                        writeCandidatesToFile=True)
    log.logger.info('Score:{}'.format(score))
    log.logger.info('top10:{}'.format(top10))
    with open('answers.txt', 'w') as f:
        f.write('\n'.join(answers))


def get_wrong_results(model_predictions, file):
    new_text_list = []

    for line in model_predictions:
        elements = line.split('\t')
        raw_head, q_text, pred_ans_id, ans_id = elements[0], elements[1], elements[2], elements[3]
        if elements[-1].strip() != '0':
            continue
        # if '##' in pred_ans_id or '##' in ans_id:
        #     continue
        head = id2entities[re.search('tensor\((\d*?),', raw_head).group(1)]
        pred_ans = id2entities[pred_ans_id]

        ans = []
        for ans_id_ in ans_id.split('##'):
            ans.append(id2entities[ans_id_])
        q_text = re.sub('^NE', '', q_text)
        new_line = head + '\t' + q_text + '[' + '##'.join(ans) + ']' + '\t' + pred_ans

        # ans = id2entities[ans_id]
        # q_text = re.sub('^NE', '', q_text)
        # new_line = head + '\t' + q_text + '\t' + ans + '\t' + pred_ans

        new_text_list.append(new_line)

    with open(os.path.join('./checkpoints', args.save_path, file), 'w') as fw:
        fw.write('\n'.join(new_text_list))


def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r', encoding='utf-8')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        d1, d2 = data_line[2].split(' ')
        data_line[2] = d1
        data_line.append(d2)
        # if no answer
        # if len(data_line) != 4:
        #     continue
        # print(data_line)
        question = data_line[0].split('[', 1)
        question_1 = question[0]
        question_2 = question[1].rsplit(']', 1)
        head = question_2[0].strip()
        question_2 = question_2[1]

        question = data_line[2] + question_1 + 'NE' + question_2

        ans = data_line[1].split('##')
        if ('' in ans) or (' ' in ans):
            print(data_line)
        data_array.append([head, question.strip(), ans, data_line[2], data_line[3]])

    if split == False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data


def get_validate_head_tails_map(valid_data_path, entity2idx):
    data_array = process_text_file(valid_data_path, split=False)
    all_triples_by_head = groupby(sorted(data_array), key=lambda x: x[0])
    head_tails_map = {}
    for num, (i, j) in enumerate(all_triples_by_head):
        # print(num)
        grouped_tails = set()
        for single_triple in list(j):
            try:
                single_head_tails = [entity2idx[item] for item in single_triple[2]]
            except:
                print('---')
            grouped_tails.update(set(single_head_tails))
        head_tails_map.update({entity2idx[i]: grouped_tails})

    return head_tails_map


def data_generator(data, dataloader, entity2idx):
    h, q_t, a_m, a, q = [], [], [], [], []

    head_embeddings, tail_embeddings, tail_ids = [], [], []

    with open(id2type_path, 'r', encoding='utf-8') as file:
        id2type = json.load(file)
    type2id = {}
    for key, value in id2type.items():
        type2id[value] = int(key)
    with open(os.path.join(args.data_path, 'complEx_schema_embedding.vec'), 'r', encoding='utf-8') as file:
        type_embedding = json.load(file)
    type_embedding_re, type_embedding_im = \
        type_embedding['ent_re_embeddings.weight'], type_embedding['ent_im_embeddings.weight']
    type_embedding_re, type_embedding_im = np.array(type_embedding_re), np.array(type_embedding_im)
    type_embeddings = np.stack([type_embedding_re, type_embedding_im], 1).reshape(len(type_embedding_re), -1)

    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0]]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            # TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        head_type, tail_type = data_sample[3], data_sample[4]
        head_id, tail_id = type2id[head_type], type2id[tail_type]
        head_embedding, tail_embedding = type_embeddings[head_id], type_embeddings[tail_id]

        h.append(head)
        q_t.append(question_tokenized)
        a_m.append(attention_mask)
        a.append(ans)
        q.append(data_sample[1])

        head_embeddings.append(head_embedding)
        tail_embeddings.append(tail_embedding)
        tail_ids.append(tail_id)

        if (i + 1) % args.batch_size == 0:
            h = torch.tensor(h, dtype=torch.long)
            q_t = torch.stack(q_t)
            a_m = torch.stack(a_m)
            head_embeddings = torch.tensor(head_embeddings, dtype=torch.float)
            tail_embeddings = torch.tensor(tail_embeddings, dtype=torch.float)
            tail_ids = torch.tensor(tail_ids, dtype=torch.long)
            yield h, q_t, a_m, a, q, head_embeddings, tail_embeddings, tail_ids
            h, q_t, a_m, a, q = [], [], [], [], []
            head_embeddings, tail_embeddings, tail_ids = [], [], []
            # yield torch.tensor(head, dtype=torch.long), question_tokenized, attention_mask, ans, data_sample[1]
    h = torch.tensor(h, dtype=torch.long)
    q_t = torch.stack(q_t)
    a_m = torch.stack(a_m)
    head_embeddings = torch.tensor(head_embeddings, dtype=torch.float)
    tail_embeddings = torch.tensor(tail_embeddings, dtype=torch.float)
    tail_ids = torch.tensor(tail_ids, dtype=torch.long)
    yield h, q_t, a_m, a, q, head_embeddings, tail_embeddings, tail_ids


hops = args.hops

model_name = args.model

# all_data_path = os.path.join(args.data_path, QA_data_path, 'all.txt')
data_path = os.path.join(args.data_path, QA_data_path, 'train.txt')
valid_data_path = os.path.join(args.data_path, QA_data_path, 'test.txt')
test_data_path = os.path.join(args.data_path, QA_data_path, 'valid.txt')

multi_hops_data = [
    os.path.join(args.data_path, 'QA', 'one_hop', 'test.txt'),
    os.path.join(args.data_path, 'QA', 'two_hop', 'test.txt'),
    os.path.join(args.data_path, 'QA', 'three_hop', 'test.txt')
]

# setting = "batch_size={}\tneg_batch_size={}\tdim={}\tlr={}\tmodel={}".format(args.batch_size, args.neg_batch_size,
#                                                                              args.relation_dim, args.lr, args.model)

# if 'webqsp' in hops:
#     data_path = '../../data/QA_data/cov19/qa_train_cov19.txt'
#     valid_data_path = '../../data/QA_data/cov19/qa_test_cov19.txt'
#     test_data_path = '../../data/QA_data/cov19/qa_test_cov19.txt'


if args.mode == 'train':
    train(data_path=data_path,
          neg_batch_size=args.neg_batch_size,
          batch_size=args.batch_size,
          shuffle=args.shuffle_data,
          num_workers=args.num_workers,
          nb_epochs=args.nb_epochs,
          embedding_dim=args.embedding_dim,
          hidden_dim=args.hidden_dim,
          relation_dim=args.relation_dim,
          gpu=args.gpu,
          use_cuda=args.use_cuda,
          valid_data_path=valid_data_path,
          patience=args.patience,
          validate_every=args.validate_every,
          freeze=args.freeze,
          hops=args.hops,
          lr=args.lr,
          entdrop=args.entdrop,
          reldrop=args.reldrop,
          scoredrop=args.scoredrop,
          l3_reg=args.l3_reg,
          model_name=args.model,
          decay=args.decay,
          ls=args.ls,
          load_from=args.load_from,
          outfile=args.outfile,
          do_batch_norm=args.do_batch_norm,
          multi_hops_data=multi_hops_data
          )
elif args.mode == 'eval':
    eval(data_path=test_data_path,
         load_from=args.load_from,
         gpu=args.gpu,
         hidden_dim=args.hidden_dim,
         relation_dim=args.relation_dim,
         embedding_dim=args.embedding_dim,
         hops=args.hops,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         model_name=args.model,
         do_batch_norm=args.do_batch_norm,
         use_cuda=args.use_cuda)

