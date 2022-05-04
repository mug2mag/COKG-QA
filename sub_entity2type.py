import json
import os
import numpy as np


with open(os.path.join('./data', 'sub_ent2type.json'), 'r', encoding='utf-8') as file:
    ent2type_ = json.load(file)

ent2type = {}
for key, value in ent2type_.items():
    key = key.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    key = key.replace('\"', '')
    key = key.replace('(2', '')
    # if key == '1.慎用：(1)血象异常者；(2)肝功能异常者；(3)肾功能不全者；(4)糖尿病患者。2.本药对胎儿有影响，妊娠妇女不宜使用。3.应用福美坦治疗时，应定期检查患者血电解质、血糖以及肝肾功能。；妊娠妇女禁用；1.对本药过敏者。2.绝经前及哺乳期妇女。':
    #     key = '1.慎用：(1)血象异常者；)肝功能异常者；(3)肾功能不全者；(4)糖尿病患者。2.本药对胎儿有影响，妊娠妇女不宜使用。3.应用福美坦治疗时，应定期检查患者血电解质、血糖以及肝肾功能。；妊娠妇女禁用；1.对本药过敏者。2.绝经前及哺乳期妇女。'
    # if key == '口服、一次4克(20丸)、一日2～3次':
    #     key = '口服、一次4克0丸)、一日2～3次'
    ent2type[key] = value

# with open(os.path.join('./data', 'entity2num.json'), 'r', encoding='utf-8') as file:
#     ent2num = json.load(file)

with open(os.path.join('./data', 'id2entity.json'), 'r', encoding='utf-8') as file:
    id2entity = json.load(file)

# with open(os.path.join('./data', 'type2id.json'), 'r', encoding='utf-8') as file:
#     type2id = json.load(file)
#
# print(type2id)

entity2id = {}
for key, value in id2entity.items():
    entity2id[value] = key

all_entity = list(id2entity.values())
print(len(all_entity))

sub_ent2type = {}
# type2ent = {}

# mat_type2ent = np.zeros((len(type2id), len(ent2num)))
# print(mat_type2ent.shape)

all_type = set()
for ent in all_entity:
    if ent == '根据不同医院，收费标准不一致，市三甲医院约10000--20000元':
        sub_ent2type[ent] = ent2type['根据不同医院，收费标准不一致，市三甲医院约10000-20000元']
    else:
        sub_ent2type[ent] = ent2type[ent]

    # sub_ent2type[ent] = ent2type[ent]

    # for t in ent2type[ent]:
    #     type_id = type2id[t]
    #     ent_id = int(entity2id[ent])
    #     mat_type2ent[type_id][ent_id] = 1
        # if t not in type2ent:
        #     type2ent[t] = [ent]
        # else:
        #     type2ent[t].append(ent)
    # if len(ent2type[ent]) > 1:
    #     print(ent)
    #     print(ent2type[ent])
    #     exit(1)

    # for t in ent2type[ent]:
    #     all_type.add(t)

# print([list(type2id)[i]+'-'+str(sum(t)) for i, t in enumerate(mat_type2ent)])
print(len(sub_ent2type))
with open(os.path.join('./data', 'sub_ent2type.json'), 'w', encoding='utf-8') as file:
    json.dump(sub_ent2type, file)
# print(all_type)
# print(len(all_type))

# type2id = {}
# for i, t in enumerate(all_type):
#     type2id[t] = i
# with open(os.path.join('../../data/one_hop_data', 'type2id.json'), 'w', encoding='gbk') as file:
#     json.dump(type2id, file)

# print(mat_type2ent.shape)
# print(sum(sum(mat_type2ent)))
#
# np.save('../../data/one_hop_data/mat_type2ent.npy', mat_type2ent)
print('根据不同医院，收费标准不一致，市三甲医院约10000--20000元#' in sub_ent2type)
