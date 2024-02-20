# /data/xfni/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/3/13 21:02
# @Author       : patrick
# @File         : calc_avg_tokens.py
# @Description  :
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
count = 0
sum_token = 0
with open('blended_skill_talk/answers.txt', 'r') as f:
    for line in f.readlines():
        sum_token += int(tokenizer(line.replace('\n', ''), return_tensors="pt")['input_ids'].shape[-1])
        count += 1
print(count)
print(sum_token)
