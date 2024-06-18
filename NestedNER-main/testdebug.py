import pickle as pkl
def load_pickle(file):
    # print("进入load_pickle")
    with open(file, 'rb') as f:
        data = pkl.load(f)
    return data

e_data = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace04/ACE04.ent_bert_base_cased_flair.emb.pkl")
e_data_1 = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace05/ACE05.ent_bert_base_cased_flair.emb.pkl")
e_data_2 = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/genia/GENIA.ent_biobert_base_cased_flair.emb.pkl")

print(e_data)
print(e_data_1)
print(e_data_2)

data = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace04/ACE04.bert_base_cased_flair.emb.pkl")
data_1 = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace05/ACE05.bert_base_cased_flair.emb.pkl")
data_2 = load_pickle(f"/mnt/LJH/XFM/NERProject/spanCL/data/datasets/genia/GENIA.biobert_base_cased_flair.emb.pkl")


print(data)
print(data_1)
print(data_2)
