import os
import json

def store_json_file(in_file, out_file, sent_start_id):
    """
    in_file: raw file, such as train.txt.clip
    out_file: json file
    """
    with open(in_file, 'r') as f:
        orig_data = json.load(f)
    new_data = []
    num_sentences = len(orig_data)
    for i in range(num_sentences):
        entities = []
        for entity in orig_data[i]["entities"]:
            b = entity["start"]
            e = entity["end"] - 1
            t = entity["type"]
            tokens = orig_data[i]["tokens"][b:e+1]
            entities.append(
                {
                    "type": t,
                    "start": b,
                    "end": e,
                    "tokens": tokens
                }
            )

        new_data.append(
            {
                "sent_id": i + sent_start_id,
                "tokens": orig_data[i]["tokens"],
                "entities": entities
            }
        )

    with open(out_file, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=1)

    print(num_sentences)

    return num_sentences

data_dir = f"//mnt/LJH/XFM/NERProject/NERData"
# digit2zero = True
save_dir = f"//mnt/LJH/XFM/NERProject/span-level/data/datasets"

for data_name in ["ace04", "ace05"]:
    print(data_name)

    train_file = f"{data_dir}/{data_name}/{data_name}_train_context.json"
    valid_file = f"{data_dir}/{data_name}/{data_name}_dev_context.json"
    test_file = f"{data_dir}/{data_name}/{data_name}_test_context.json"

    # os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
    train_save_file = f"{save_dir}/{data_name}/train.json"
    valid_save_file = f"{save_dir}/{data_name}/valid.json"
    test_save_file = f"{save_dir}/{data_name}/test.json"

    num_train_sentences = store_json_file(
        train_file, train_save_file,
        sent_start_id=0,
    )
    num_valid_sentences = store_json_file(
        valid_file, valid_save_file,
        sent_start_id=num_train_sentences,
    )
    num_test_sentences = store_json_file(
        test_file, test_save_file,
        sent_start_id=num_train_sentences + num_valid_sentences,
    )

def split_genia_train_dev(source_file_path,train_file_path,dev_file_path,train_percentage):
    # 读取原始json文件数据
    with open(source_file_path, 'r') as f:
        data = json.load(f)
    # 计算切分点 并按照要求进行切分
    split_index = int(len(data) * train_percentage)
    train_data, dev_data = data[:split_index], data[split_index:]
    # 写入两个新的json文件
    with open(train_file_path, 'w') as f1:
        json.dump(train_data,  f1, ensure_ascii=False, indent=1)
    with open(dev_file_path, 'w') as f2:
        json.dump(dev_data,  f2, ensure_ascii=False, indent=1)
for data_name in ["genia"]:
    print(data_name)
    # 把train_dev数据集按9：1分开
    train_dev_file = f"{data_dir}/{data_name}/{data_name}_train_dev_context.json"
    train_file = f"{data_dir}/{data_name}/{data_name}_train_context.json"
    dev_file = f"{data_dir}/{data_name}/{data_name}_dev_context.json"
    split_genia_train_dev(train_dev_file,train_file,dev_file,0.9)

    test_file = f"{data_dir}/{data_name}/{data_name}_test_context.json"
    # os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
    train_save_file = f"{save_dir}/{data_name}/train.json"
    valid_save_file = f"{save_dir}/{data_name}/valid.json"
    test_save_file = f"{save_dir}/{data_name}/test.json"

    num_train_sentences = store_json_file(
        train_file, train_save_file,
        sent_start_id=0,
    )
    num_valid_sentences = store_json_file(
        valid_file, valid_save_file,
        sent_start_id=num_train_sentences,
    )
    num_test_sentences = store_json_file(
        test_file, test_save_file,
        sent_start_id=num_train_sentences + num_valid_sentences,
    )