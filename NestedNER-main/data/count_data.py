import os
import json

data_dir = f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets"


# 统计每个数据集的句子总数、实体总数、word总数
def count_numbers(file_path, data_name):
    with open(file_path, 'r') as f:
        data = json.load(f)
    num_sentences = len(data)
    data_counting = {}
    num_entities = 0
    num_words = 0

    max_size_sent = sum_size_sent = avg_size_sent = 0
    max_size_entity = sum_size_entitiy = avg_size_entity = 0
    max_size_word = sum_size_word = avg_size_word = 0
    for i in range(num_sentences):
        num_words += len(data[i]['tokens'])
        num_entities += len(data[i]['entities'])
        max_size_sent = max(max_size_sent, len(data[i]['tokens']))
        sum_size_sent +=len(data[i]['tokens'])
        for token in data[i]['tokens']:
            if max_size_word <len(token):
                max_size_word = len(token)
            # max_size_word = max(max_size_word, len(token))
            sum_size_word+=len(token)
        for entity in data[i]['entities']:
            max_size_entity = max(max_size_entity, len(entity["tokens"]))
            sum_size_entitiy+=len(entity["tokens"])
    avg_size_sent = float(sum_size_sent)/float(num_sentences)
    avg_size_entity =float(sum_size_entitiy)/float(num_entities)
    avg_size_word = float(sum_size_word)/float(num_words)


    data_counting["dataset_name"] = data_name

    data_counting["num_sentences"] = num_sentences
    data_counting["num_entities"] = num_entities
    data_counting["num_words"] = num_words

    data_counting["avg_size_sent"] =avg_size_sent
    data_counting["avg_size_entity"] =avg_size_entity
    data_counting["avg_size_word"]=avg_size_word

    data_counting["max_size_sent"] = max_size_sent
    data_counting["max_size_entity"] = max_size_entity
    data_counting["max_size_word"] = max_size_word
    return data_counting


for data_name in ["ace04", "ace05", "genia"]:
    print(data_name)

    train_file = f"{data_dir}/{data_name}/train.json"
    valid_file = f"{data_dir}/{data_name}/valid.json"
    test_file = f"{data_dir}/{data_name}/test.json"

    num_train = count_numbers(train_file, "train")
    print(num_train)
    num_valid = count_numbers(valid_file, "valid")
    print(num_valid)
    num_test = count_numbers(test_file, "test")
    print(num_test)
