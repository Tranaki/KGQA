import json

# 加载关系和关系ID之间的映射关系
with open('../data/wiki80/wiki80_rel2id.json', 'r') as f:
    rel2id = json.load(f)

# 处理实体和关系
entity_id = 0
rel_id = 0
entity2id = {}
rel2id_new = {}
# with open('../data/wiki80/wiki80_train.txt', 'r') as f:
with open('../data/wiki80/wiki80_val.txt', 'r') as f:
    for line in f:
        data = json.loads(line)
        head = data['h']['id']
        tail = data['t']['id']
        rel = data['relation']
        if head not in entity2id:
            entity2id[head] = entity_id
            entity_id += 1
        if tail not in entity2id:
            entity2id[tail] = entity_id
            entity_id += 1
        if rel not in rel2id_new:
            rel2id_new[rel] = rel_id
            rel_id += 1

# 处理数据
# with open('../data/wiki80/wiki80_train_processed.txt', 'w') as f:
    # with open('../data/wiki80/wiki80_train.txt', 'r') as f2:
with open('../data/wiki80/wiki80_val_processed.txt', 'w') as f:
    with open('../data/wiki80/wiki80_val.txt', 'r') as f2:
        for line in f2:
            data = json.loads(line)
            head = entity2id[data['h']['id']]
            tail = entity2id[data['t']['id']]
            rel = rel2id_new[data['relation']]
            input_seq = '{} {} {}'.format(head, rel, tail)
            output_seq = str(tail)
            f.write('{}\t{}\n'.format(input_seq, output_seq))
