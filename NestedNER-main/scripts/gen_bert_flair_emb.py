import numpy as np
import argparse
import json
import pickle
from tqdm import tqdm

from flair.embeddings import BertEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Token, Sentence

def form_sentence(tokens):
    s = Sentence(" ".join(tokens))
    if args.need_cls_emb:
        cls_token = Token("[CLS]")
        s.tokens.insert(0, cls_token)
    return s

def get_embs(s):
    ret = []
    for t in s:
        ret.append(t.get_embedding().cpu().numpy())
    return np.stack(ret, axis=0)

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


parser = argparse.ArgumentParser(description='Arguments for training.')
parser.add_argument('--data_dir', default='ACE05', type=str)
parser.add_argument('--model_name', default='bert-large-cased', type=str)
parser.add_argument('--lm_emb_save_path', default='lm.emb.pkl', type=str)
parser.add_argument('--cased', default=0, type=int)
parser.add_argument('--add_flair', default=0, type=int)
parser.add_argument('--flair_name', default='news', type=str)
parser.add_argument('--need_cls_emb', default=1, type=int)

args = parser.parse_args()

# export CUDA_VISIBLE_DEVICES=3
#
# # ACE04
args.data_dir = "/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace04"
args.model_name = "bert-large-uncased"
args.flair_name = "news"
args.lm_emb_save_path = f"{args.data_dir}/ACE04.bert_large_uncased_flair.emb.pkl"
args.cased = 0
args.add_flair = 1
args.need_cls_emb = 1

# args.data_dir = "/mnt/LJH/XFM/NERProject/spanCL/data/datasets/ace04"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.lm_emb_save_path = f"{args.data_dir}/ACE04.bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

# ACE05
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/ace05"
# args.model_name = "bert-large-uncased"
# args.flair_name = "news"
# args.lm_emb_save_path = f"{args.data_dir}/ACE05.bert_large_uncased_flair.emb.pkl"
# args.cased = 0
# args.add_flair = 1
# args.need_cls_emb = 1

# args.data_dir = f"//mnt/LJH/XFM/NERProject/span-level/data/datasets/ace05"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/ACE05.bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1


# GENIA
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/genia"
# args.model_name = "dmis-lab/biobert-large-cased-v1.1"
# args.flair_name = "pubmed"
# args.lm_emb_save_path = f"{args.data_dir}/GENIA.biobert_large_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1
# args.need_cls_emb = 1

# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/genia"
# args.model_name = "dmis-lab/biobert-base-cased-v1.1"
# args.flair_name = "pubmed"
# args.ent_lm_emb_save_path = f"{args.data_dir}/GENIA.biobert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

# NNE
# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/nne"
# args.model_name = "bert-large-uncased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/NNE.bert_large_uncased_flair.emb.pkl"
# args.cased = 0
# args.add_flair = 1

# args.data_dir = "/NAS2020/Workspaces/NLPGroup/juncw/database/NER/nested_ner/nne"
# args.model_name = "bert-base-cased"
# args.flair_name = "news"
# args.ent_lm_emb_save_path = f"{args.data_dir}/NNE.bert_base_cased_flair.emb.pkl"
# args.cased = 1
# args.add_flair = 1

bert_embedding = BertEmbeddings(args.model_name, layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")
if args.add_flair:
    flair_embedding = StackedEmbeddings([
        FlairEmbeddings(f'{args.flair_name}-forward'),
        FlairEmbeddings(f'{args.flair_name}-backward'),
    ])

if args.cased:
    bert_embedding.tokenizer.basic_tokenizer.do_lower_case = False

train = load_json(f"{args.data_dir}/train.json")
valid = load_json(f"{args.data_dir}/valid.json")
test = load_json(f"{args.data_dir}/test.json")
dataset = train + valid + test
    
emb_dict = {}
for item in tqdm(dataset):
    tokens = tuple(item['tokens'])#list 8
    s = form_sentence(tokens)# Sentence 9
    
    s.clear_embeddings()
    bert_embedding.embed(s)
    emb = get_embs(s) # (T, 4*H) 9*768

    cls_emb = None
    if args.need_cls_emb:
        cls_emb = emb[0, :]#768
        tokens = tuple(item['tokens'])
        s = form_sentence(tokens)
        emb = emb[1:, :]#8*768

    if args.add_flair:
        s.clear_embeddings()
        flair_embedding.embed(s)
        if args.need_cls_emb:
            g_emb = get_embs(s.tokens[1:])#8*4096
        else:
            g_emb = get_embs(s)
        emb = np.concatenate([emb, g_emb], axis=-1)#8*4864
    
    emb_dict[tokens] = {
        "word_emb": emb.astype('float16'),
        "cls_emb": cls_emb.astype('float16'),
    }
    
with open(args.lm_emb_save_path, 'wb') as f:
    pickle.dump(emb_dict, f)