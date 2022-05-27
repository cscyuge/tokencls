import pickle
import torch
import numpy as np
from tqdm import tqdm
import re
from transformers import MT5Tokenizer, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import os
from pprint import pprint
import argparse
import json
from utils.kmp import KMP
kmp = KMP()
CLS = '▁'
SEP = '</s>'

bert_model = 'hfl/chinese-bert-wwm-ext'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

bert_model = 'google/mt5-small'
tokenizer = MT5Tokenizer.from_pretrained(bert_model)


def preprocess(dataset_all,path_to, index):
    srcs = []
    contents_list = []
    for data in dataset_all:
        contents = data['contents']
        srcs.append(data['src'])
        contents_list.append(contents)

    src_ids, tar_ids = [], []
    src_tokens, tar_labels = [], []
    worker_num = 1

    dataset = []
    for i in range(worker_num):
        if i == index:
            for j in range(len(srcs[i * len(srcs) // worker_num: (i + 1) * len(srcs) // worker_num])):
                dataset.append((srcs[i * len(srcs) // worker_num + j], contents_list[i * len(srcs) // worker_num + j]))

    for src, contents in tqdm(dataset,ascii=True, ncols=50):
        src = src.strip()
        src = 'output the key words: ' + src
        # print(src)
        tokens = tokenizer.tokenize(re.sub('\*\*', '', src).lower())
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)

        tokens = []
        masks = []
        # tokens = [CLS] + tokenizer.tokenize(re.sub('\*\*', '', src).lower()) + [SEP]
        _keywords = []

        for content in contents:
            src = content['text']
            _src_tokens = bert_tokenizer.tokenize(re.sub('\*\*', '', src).lower())
            src_masks = np.array([0 for _ in range(len(_src_tokens))])
            for tooltip in content['tooltips']:
                key = tooltip['origin']
                keyword_tokens = bert_tokenizer.tokenize(key)
                i = kmp.kmp(_src_tokens, keyword_tokens)
                l = len(keyword_tokens)
                if i != -1:
                    src_masks[i] = 1
                    if l >= 2:
                        # print("get keyword token len >= 2")
                        src_masks[i + l - 1] = 4
                        src_masks[i + 1:i + l - 1] = 3
                    _keywords.append(key)
            tokens += _src_tokens
            masks += list(src_masks)
        src_tokens.append(tokens)
        tar_labels.append(masks)

        tar = '，'.join(_keywords)

        tokens = tokenizer.tokenize(re.sub('\*\*', '', tar).lower())
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tar_ids.append(ids)


    print(len(src_ids))
    src_ids_smaller, tar_ids_smaller, src_tokens_smaller, tar_labels_smaller = [], [], [], []
    max_len = 512
    for src, tar, tokens, labels in zip(src_ids, tar_ids, src_tokens, tar_labels):
        if len(src) < max_len and len(src) > 2:
            src_ids_smaller.append(src)
            tar_ids_smaller.append(tar)
            src_tokens_smaller.append(tokens)
            tar_labels_smaller.append(labels)

    src_ids, tar_ids,src_tokens, tar_labels = src_ids_smaller, tar_ids_smaller, src_tokens_smaller, tar_labels_smaller
    print(len(src_ids))

    src_ids = pad_sequences(src_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
    tar_ids = pad_sequences(tar_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]
    tar_masks = [[float(i != 0.0) for i in ii] for ii in src_masks]

    with open(os.path.join(path_to, 'src_ids.pkl'), 'wb') as f:
        pickle.dump(src_ids, f)
    with open(os.path.join(path_to, 'tar_ids.pkl'), 'wb') as f:
        pickle.dump(tar_ids, f)
    with open(os.path.join(path_to, 'src_masks.pkl'), 'wb') as f:
        pickle.dump(src_masks, f)
    with open(os.path.join(path_to, 'tar_masks.pkl'), 'wb') as f:
        pickle.dump(tar_masks, f)
    with open(os.path.join(path_to, 'src_tokens.pkl'), 'wb') as f:
        pickle.dump(src_tokens, f)
    with open(os.path.join(path_to, 'tar_labels.pkl'), 'wb') as f:
        pickle.dump(tar_labels, f)


def main(index):
    dataset = json.load(open('../../data/dataset_new_3.json', 'r', encoding='utf-8'))
    total = len(dataset)
    print('train dataset:')
    preprocess(dataset[:int(total / 10 * 8)], './data/train', index)
    print('test dataset:')
    preprocess(dataset[int(total / 10 * 8):int(total / 10 * 9)], './data/test', index)
    print('valid dataset:')
    preprocess(dataset[int(total / 10 * 9):], './data/valid', index)
    print('done')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', type=int)
    # args = parser.parse_args()
    # print(args.i)
    # main(args.i)
    main(0)


