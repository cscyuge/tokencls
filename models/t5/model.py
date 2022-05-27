import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from importlib import import_module
from tqdm import tqdm
import copy
from utils.bleu_eval import count_score
from utils.dataset_mt5 import build_dataset, build_iterator
from transformers import MT5ForConditionalGeneration, BertTokenizer
from sklearn.metrics import f1_score, accuracy_score
import shutil

CLS = '▁'
SEP = '</s>'


def build(batch_size, cuda):

    x = import_module('config')
    pretrained_model = 'google/mt5-small'
    config = x.Config(batch_size, pretrained_model)
    train_data = build_dataset(config, './data/train/src_ids.pkl', './data/train/src_masks.pkl',
                               './data/train/tar_ids.pkl', './data/train/tar_masks.pkl', './data/train/tar_labels.pkl',
                               './data/train/src_tokens.pkl')
    test_data = build_dataset(config, './data/test/src_ids.pkl', './data/test/src_masks.pkl',
                              './data/test/tar_ids.pkl', './data/test/tar_masks.pkl', './data/test/tar_labels.pkl',
                              './data/test/src_tokens.pkl')
    val_data = build_dataset(config, './data/valid/src_ids.pkl', './data/valid/src_masks.pkl',
                             './data/valid/tar_ids.pkl', './data/valid/tar_masks.pkl', './data/valid/tar_labels.pkl',
                             './data/valid/src_tokens.pkl')
    train_dataloader = build_iterator(train_data, config)
    val_dataloader = build_iterator(val_data, config)
    test_dataloader = build_iterator(test_data, config)

    model = MT5ForConditionalGeneration.from_pretrained(pretrained_model)

    model = model.to(config.device)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate
    )

    return model, optimizer, train_dataloader, val_dataloader, test_dataloader, config


def valid(model, dataloader, config):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    predictions, true_labels = [], []
    results = []
    valid_bert_model = 'hfl/chinese-bert-wwm-ext'
    valid_tokenizer = BertTokenizer.from_pretrained(valid_bert_model)
    with open('./result/label_test.txt','w',encoding='utf-8') as f:
        for i, (batch_src, batch_tar, batch_labels) in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                outputs = model(input_ids=batch_src[0], attention_mask=batch_src[1], labels=batch_tar[0])
                loss = outputs.loss
                eval_loss += loss.item()
                outputs = model.generate(input_ids=batch_src[0], attention_mask=batch_src[1], do_sample=False)
                batch_keywords = config.tokenizer.batch_decode(outputs,skip_special_tokens=True)
                results += batch_keywords
                for labels in batch_labels:
                    true_labels += labels
                for tokens, keywords, labels in zip(batch_src[2], batch_keywords, batch_labels):
                    keywords = keywords.replace('，', ',').split(',')
                    print(keywords)
                    keywords = [keyword.strip() for keyword in keywords]
                    masks = np.array([0 for _ in range(len(tokens))])
                    if len(masks) != len(labels):
                        print(len(tokens), tokens)
                        print(len(labels),labels)
                    for keyword in keywords:
                        keyword_tokens = valid_tokenizer.tokenize(keyword)
                        l = len(keyword_tokens)
                        for j in range(len(tokens) - l):
                            if tokens[j:j + l] == keyword_tokens:
                                masks[j:j + l] = 1
                                if l >= 2:
                                    # print("get keyword token len >= 2")
                                    masks[j + l - 1] = 4
                                    masks[j + 1:j + l - 1] = 3
                                break
                    predictions += masks.tolist()
                    for token, mask, label in zip(tokens, list(masks), labels):
                        f.write(token+' '+str(mask)+' '+str(label)+'\n')
                    f.write('\n')

            if i % 100 == 0:
                print('eval loss:%f' % loss.item())
    eval_loss /= len(dataloader)
    print(results[:20])
    acc = accuracy_score(predictions, true_labels)
    f1 = f1_score(predictions, true_labels, average='micro')
    return acc, f1, eval_loss


def train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config):
    #training steps
    max_f1 = -99999
    save_file = {}
    for e in range(config.num_epochs):
        model.train()
        train_loss = 0
        for i, (batch_src, batch_tar, batch_labels) in tqdm(enumerate(train_dataloader)):
            model_outputs = model(input_ids=batch_src[0], attention_mask=batch_src[1], labels=batch_tar[0])
            loss = model_outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 1000 == 0:
                print('train loss:%f' %loss.item())
        train_loss /= len(train_dataloader)
        print("Train loss: {}".format(train_loss))

        #validation steps
        acc, f1, val_loss = valid(model, val_dataloader, config)
        print("Valid loss: {}".format(val_loss))
        print("Validation Accuracy: {}".format(acc))
        print("Validation F1-Score: {}".format(f1))
        # if f1 > max_f1:
        if True:
            max_f1 = f1
            save_file['epoch'] = e + 1
            save_file['para'] = model.state_dict()
            save_file['best_acc'] = acc
            save_file['best_f1'] = f1
            torch.save(save_file, './cache/best_save.data')
            shutil.copy('result/label_test.txt', 'result/label_test_best.txt')

        print(save_file['epoch'] - 1)

    save_file_best = torch.load('./cache/best_save.data')
    print('Train finished')
    print('Best Val acc:%f' % (save_file_best['best_acc']))
    print('Best val epoch:', save_file_best['epoch'])
    model.load_state_dict(save_file_best['para'])
    acc, f1, test_loss = valid(model, test_dataloader, config)
    print("Test loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(acc))
    print("Test F1-Score: {}".format(f1))




def main():
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, config = build(1, True)
    train(model, optimizer, train_dataloader, val_dataloader, test_dataloader, config)
    # print('finish')


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
