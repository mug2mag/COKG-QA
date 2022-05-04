import json
import re
import random
from tqdm import tqdm
import copy
# 加载的数据为整理好的三元组list

def save_word2id(name_set, data_type, save_name):
    word2id = {name:id for id, name in enumerate(name_set)}
    id2word = {id:name for id, name in enumerate(name_set)}

    with open("benchmarks/{}/{}2id.txt".format(save_name, data_type), "w") as f:
        f.write('{}'.format(len(word2id)))
        for word in word2id:
            f.write('\n{}\t{}'.format(word, word2id[word]))
    return word2id, id2word
def save_id2word(id2entity, type, save_name):
    with open('benchmarks/{}/id2entity.json'.format(save_name, type), 'w') as fw:
        json.dump(id2entity, fw, ensure_ascii=False, indent=2)
def save_2id_txt(triples_list, data_type, save_name):
    with open("benchmarks/{}/{}2id.txt".format(save_name, data_type), "w") as f:
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
def loadJson(location):
    print("location:", location)
    with open(location,'r') as load_f:
        return json.load(load_f)
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
    dataset_location = '{}{}_KG.json'.format(SourceKgLocation, dataset)
    entities_set = set()
    relations_set = set()
    word_triples = []
    print("load data from {} ...".format(dataset_location))
    load_f = loadBigJson(dataset_location)
    for triple in tqdm(load_f):
        head = triple[0]
        tail = triple[2].replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        relation = triple[1]
        if not all([head, tail, relation]):
            continue
        entities_set.add(head)
        entities_set.add(tail)
        relations_set.add(relation)
        # print(head, relation, tail)
        word_triples.append([head, relation, tail])

    entities2id, id2entities = save_word2id(entities_set, 'entity', save_name)
    relations2id, id2relations = save_word2id(relations_set, 'relation', save_name)
    saveJson(id2entities, '../../../data/COKG_data/id2entity.json')
    # entity2num = get_entity_num(word_triples, entities2id)


    save_id2word(id2entities, 'entity', save_name)

    triples_num = len(word_triples)
    valid_num = triples_num // 5
    test_num = valid_num
    train_num = triples_num - 2 * valid_num

    id_triples = []
    for triple in word_triples:
        head = triple[0]
        tail = triple[2]
        relation = triple[1]
        id_triples.append([entities2id[head], relations2id[relation], entities2id[tail]])

    random.shuffle(id_triples)
    train_triples = copy.deepcopy(id_triples)
    valid_triples = copy.deepcopy(id_triples[train_num:train_num + valid_num])
    test_triples = copy.deepcopy(id_triples[train_num + valid_num:triples_num])


    train_triples_ = []
    for triple in train_triples:
        train_triples_.append(triple)
    random.shuffle(train_triples_)
    random.shuffle(test_triples)
    random.shuffle(valid_triples)

    print('train: ', len(train_triples_))
    print('test: ', len(test_triples))
    print('valid: ', len(valid_triples))
    save_2id_txt(train_triples_, 'train', save_name)
    # save_2id_txt(train_triples, 'train', save_name)
    save_2id_txt(valid_triples, 'valid', save_name)
    save_2id_txt(test_triples, 'test', save_name)

    with open('benchmarks/{}/id2entity.json'.format(save_name), 'w') as fw:
        json.dump(id2entities, fw, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    datasets = ['gethered']
    for dataset in datasets:
        main(dataset)
