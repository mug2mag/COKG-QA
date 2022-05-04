import json
import re
import random
from tqdm import tqdm
import argparse



# 加载的数据为整理好的三元组list

def save_word2id(name_set, data_type):
    word2id = {name: id for id, name in enumerate(name_set)}
    id2word = {id: name for id, name in enumerate(name_set)}

    with open("benchmarks/gethered_type/{}2id.txt".format(data_type), "w") as f:
        f.write('{}'.format(len(word2id)))
        for word in word2id:
            f.write('\n{}\t{}'.format(word, word2id[word]))
    return word2id, id2word
def save_id2word(id2entity):
    with open('benchmarks/gethered_type/id2entity.json', 'w') as fw:
        json.dump(id2entity, fw, ensure_ascii=False, indent=2)
def save_2idsave_id2word_txt(triples_list, data_type):
    with open("benchmarks/gethered_type/{}2id.txt".format(data_type), "w") as f:
        f.write('{}'.format(len(triples_list)))
        for triple in triples_list:
            f.write('\n{} {} {}'.format(triple[0], triple[2], triple[1]))
def save_2id_txt(triples_list, data_type):
    with open("benchmarks/gethered_type/{}2id.txt".format(data_type), "w") as f:
        f.write('{}'.format(len(triples_list)))
        for triple in triples_list:
            f.write('\n{} {} {}'.format(triple[0], triple[2], triple[1]))
def loadBigJson(location):
    f = []
    with open(location, "r") as fr:
        line = fr.readline()
        while line:
            try:
                line = json.loads(line)
            except:
                print(line)
            # if random.randint(0, 1000) == 1:
            #     f.append(line)
            f.append(line)
            line = fr.readline()
    return f
def saveJson(file, location):
    with open(location, 'w') as fw:
        json.dump(file, fw, ensure_ascii=False, indent=2)


def get_entity_num(triples, entity2id):
    entity_num = [0] * len(entity2id)
    for triple in triples:
        head = triple[0]
        tail = triple[2]
        entity_num[entity2id[head]] += 1
        entity_num[entity2id[tail]] += 1
    return entity_num


def main(dataset):
    SourceKgLocation = '../../../data/COKG_data/KG/'
    save_name = dataset
    entities_set = set()
    relations_set = set()
    word_triples = []
    KG_with_type = loadBigJson('{}{}_KG_type.json'.format(SourceKgLocation, dataset))
    tail_entity2domain = {}
    entity2type = {}
    for triple in tqdm(KG_with_type):
        head = triple[3][0]
        tail = triple[3][1]
        relation = triple[1]
        if not all([head, tail, relation]):
            continue
        entities_set.add(head)
        entities_set.add(tail)
        relations_set.add(relation)
        # print(head, relation, tail)
        if [head, relation, tail] not in word_triples:
            word_triples.append([head, relation, tail])
        if triple[2] not in tail_entity2domain:
            tail_entity2domain.update({triple[2] : []})
        if tail not in tail_entity2domain[triple[2]]:
            tail_entity2domain[triple[2]].append(tail)

        if triple[0] not in entity2type:
            entity2type.update({triple[0] : []})
        if triple[3][0] not in entity2type[triple[0]]:
            entity2type[triple[0]].append(tail)
        if triple[2] not in entity2type:
            entity2type.update({triple[2] : []})
        if triple[3][1] not in entity2type[triple[2]]:
            entity2type[triple[2]].append(tail)

    saveJson(tail_entity2domain, '../../../data/COKG_data/tail_entity2domain.json')
    saveJson(entity2type, '../../../data/COKG_data/sub_ent2type.json')

    entities2id, id2entities = save_word2id(entities_set, 'entity')
    relations2id, id2relations = save_word2id(relations_set, 'relation')
    entity2num = get_entity_num(word_triples, entities2id)

    KG_triples = []
    for triple in word_triples:
        # head_num = entity2num[entities2id[triple[0]]]
        # tail_num = entity2num[entities2id[triple[2]]]
        KG_triples.append(triple)
        # if min(head_num, tail_num) < 50:
        #     copy_times = 50 - min(head_num, tail_num)
        #     for i in range(copy_times):
        KG_triples.append(triple)

    saveJson(id2entities, '../../../data/COKG_data/id2type.json')
    save_id2word(id2entities)

    triples_num = len(KG_triples)
    valid_num = triples_num // 5
    test_num = valid_num
    train_num = triples_num - 2 * valid_num

    id_triples = []
    for triple in KG_triples:
        head = triple[0]
        tail = triple[2]
        relation = triple[1]
        id_triples.append([entities2id[head], relations2id[relation], entities2id[tail]])

    random.shuffle(id_triples)
    train_triples = id_triples[:train_num].copy()
    valid_triples = id_triples[train_num:train_num + valid_num].copy()
    test_triples = id_triples[train_num + valid_num:triples_num].copy()

    save_2id_txt(id_triples, 'train')
    # save_2id_txt(train_triples, 'train')
    save_2id_txt(valid_triples, 'valid')
    save_2id_txt(test_triples, 'test')

    with open('benchmarks/gethered_type/id2entity.json', 'w') as fw:
        json.dump(id2entities, fw, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gethered')
    args = parser.parse_args()

    dataset = args.dataset
    main(dataset)
