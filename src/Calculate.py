# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2023/4/22 17:10
# @Author : Patrick Ni
# @File : Calculate.py
# @description:
import pandas as pd

from load_models_and_datasets import load_prompt_datasets
from transformers import AutoTokenizer


def trans_str(string):
    string = string.replace('["none"]', '')
    strings = string.split(']')
    lists = []
    for s in strings:
        if len(s) <= 1 or s[0] != '[':
            continue
        s = eval(s + ']')
        lists.extend(s)
    return lists


def calculate_dataset_properties():
    tokenizer = AutoTokenizer.from_pretrained('~/code/Pretrained_Models/flan-t5-xxl')
    x = pd.read_csv('~/code/Datasets/atomic/v4_atomic_all.csv')
    # Index(['event', 'oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent',
    #        'xNeed', 'xReact', 'xWant', 'prefix', 'split'],
    intent = x[x['xIntent'] != '[]']
    need = x[x['xNeed'] != '[]']
    intent = intent.groupby(by='event').sum()
    intent = intent.reset_index('event')
    need = need.groupby(by='event').sum()
    need = need.reset_index('event')
    intent = intent.loc[:, ['event', 'xIntent']]
    need = need.loc[:, ['event', 'xNeed']]
    # print("go_emotion: ", len(GoEmotions), check_token_len(tokenizer, GoEmotions['text'])[1], 1)
    # print("wow: ", len(wow['text']), check_token_len(tokenizer, wow['text'])[1],
    #       check_token_len(tokenizer, wow['knowledge'])[1])
    # print("persona: ", len(process_persona), check_token_len(tokenizer, process_text)[1],
    #       check_token_len(tokenizer, process_persona)[1])
    xneed, xintent = [], []
    for x in need['xNeed']:
        xneed.extend(trans_str(x))
    for x in intent['xIntent']:
        xintent.extend(trans_str(x))
    print("event: ", len(intent), check_token_len(tokenizer, intent['event'])[1],
          check_token_len(tokenizer, xintent)[1],
          check_token_len(tokenizer, need['event'])[1], check_token_len(tokenizer, xneed)[1])
    # print("topic: ", len(process_dd), check_token_len(tokenizer, process_dd)[1],
    #       check_token_len(tokenizer, process_topic)[1]),


def check_token_len(tokenizer, texts):
    max_len = 0
    sum_len = 0
    count = len(texts)
    tag = 1
    for t in texts:
        token_len = tokenizer(t, return_tensors="pt", truncation=True, padding=True,
                              add_special_tokens=True)['input_ids'].shape[-1]
        if tag == 1:
            print(t)
            print(token_len)
            tag = 0
        sum_len += token_len
        max_len = max_len if max_len > token_len else token_len
    return max_len, sum_len / count, count


if __name__ == '__main__':
    calculate_dataset_properties()
