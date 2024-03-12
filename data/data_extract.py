# /user/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Patrick
# @Time: 2022/10/09 22:44
# @File: data_extract.py
# description:
import pandas as pd
import jsonlines

word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
              "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
              "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
              "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
              "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}


def clean(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    return sentence


# count = 0
# data = pd.read_json(f'/data/xfni/code/PMDialogueSystem/data/empathetic/parsed_emotion_Ekman_intent_train.json')
# writer = open('empathetic/train.txt', 'w')
# utter = data['utterence']
# for x in utter:
#     count += 1
#     line = ''
#     for y in range(len(x) - 1):
#         line = line + x[y] + ' '
#     ans = x[-1]
#     writer.write(line + ans + '\n')
#     with jsonlines.open("empathetic/train.jsonl", mode='a') as w:
#         train_data = {'text': line, 'summary': ans}
#         w.write(train_data)
# writer.close()
# print(count)
count = 0
data = pd.read_json(f'/data/xfni/code/PMDialogueSystem/data/empathetic/parsed_emotion_Ekman_intent_valid.json')
writer = open('empathetic/val.txt', 'w')
utter = data['utterence']
for x in utter:
    count += 1
    line = ''
    for y in range(len(x) - 1):
        line = line + x[y] + ' '
    ans = x[-1]
    writer.write(line + ans + '\n')
    with jsonlines.open("empathetic/val.jsonl", mode='a') as w:
        train_data = {'text': line, 'summary': ans}
        w.write(train_data)
writer.close()
print(count)

count = 0
data = pd.read_json(f'/data/xfni/code/PMDialogueSystem/data/empathetic/parsed_emotion_Ekman_intent_valid.json')
writer = open('empathetic/val.txt', 'w')
utter = data['utterence']
for x in utter:
    count += 1
    line = ''
    for y in range(len(x) - 1):
        line = line + x[y] + ' '
    ans = x[-1]
    writer.write(line + ans + '\n')
    with jsonlines.open("empathetic/val.jsonl", mode='a') as w:
        train_data = {'text': line, 'summary': ans}
        w.write(train_data)
writer.close()
print(count)
