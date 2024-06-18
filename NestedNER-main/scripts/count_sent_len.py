import json

def count_sent_len(datas_json_file, label):
    with open(datas_json_file, 'r') as f:
        datas = json.load(f)

    sent_num = len(datas)

    max_sent_len = max([len(d["tokens"]) for d in datas])
    total_sent_len = sum([len(d["tokens"]) for d in datas])
    num_ent = sum([len(d["entities"]) for d in datas])

    num_nested_ent = 0
    for d in datas:
        n_ent = len(d["entities"])
        for i in range(n_ent):
            i_start, i_end = d["entities"][i]["start"], d["entities"][i]["end"]
            for j in range(n_ent):
                if j == i:
                    continue
                j_start, j_end = d["entities"][j]["start"], d["entities"][j]["end"]
                if j_start <= i_start and j_end >= i_start:
                    num_nested_ent += 1
                    break
                elif j_start <= i_end and j_end >= i_end:
                    num_nested_ent += 1
                    break
                elif j_start >= i_start and j_end <= i_end:
                    num_nested_ent += 1
                    break
    avg_sent_len = round(total_sent_len/sent_num, 1)
    nested_ent_ratio = round(100 * num_nested_ent/num_ent, 1)

    print(label,'\t','\t',sent_num,'\t', '\t',avg_sent_len,'\t','\t', max_sent_len,'\t','\t', num_ent,'\t', '\t',num_nested_ent,'\t', '\t',nested_ent_ratio)


if __name__ == "__main__":
    data_dir = f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/"
    data_names = [
        # "ace04",
        # "msra", "people_daily", "ecommerce",
        # "weibo", "resume", "ontonote4",  # Chinese flat NER dataset
        # "scierc", "conll04",  # entity-relation dataset
        "ace04", "ace05", "genia",
        # "nne", # nested NER dataset
        # "conll03",  # English flat NER dataset
    ]

    for data_name in data_names:
        print(f"\n{data_name}\tsent_num\tavg_sent_len\tmax_sent_len\tnum_ent\tnum_nested_ent\tnested_ent_ratio")

        count_sent_len(f"{data_dir}/{data_name}/train.json", "train")
        count_sent_len(f"{data_dir}/{data_name}/valid.json", "valid")
        count_sent_len(f"{data_dir}/{data_name}/test.json", "test")



