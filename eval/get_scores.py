# /user/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Patrick
# @Time: 2022/10/10 14:37
# @File: get_scores.py
# description:
# import pandas as pd
import json

with open('eval_results/odkg_raw_topk_results.json', 'r') as f:
    data = json.load(f)

ppl_min, p_id = 10000, -1
rougel_max, r_id = 0, -1
bertscore_max, b_id = 0, -1
meteor_max, m_id = 0, -1
d1_max, d1_id = 0, -1
d2_max, d2_id = 0, -1
bleu1_max, b1_id = 0, -1
bleu4_max, b4_id = 0, -1
cider_max, c_id = 0, -1
for i in range(len(data)):
    score = data[i]
    ppl = score['ppl']
    rougel = score['ROUGE_L']
    bertscore = score['bertscore']
    meteor = score['METEOR']
    bleu1 = score['Bleu_1']
    bleu4 = score['Bleu_4']
    cider = score['CIDEr']
    d1 = score['d1']
    d2 = score['d2']
    if ppl < ppl_min:
        ppl_min = ppl
        p_id = i
    if rougel > rougel_max:
        rougel_max = rougel
        r_id = i
    if bertscore > bertscore_max:
        bertscore_max = bertscore
        b_id = i
    if meteor > meteor_max:
        meteor_max = meteor
        m_id = i
    if bleu1 > bleu1_max:
        bleu1_max = bleu1
        b1_id = i
    if bleu4 > bleu4_max:
        bleu4_max = bleu4
        b4_id = i
    if cider > cider_max:
        cider_max = cider
        c_id = i
    if d1 > d1_max:
        d1_max = d1
        d1_id = i
    if d2 > d2_max:
        d2_max = d2
        d2_id = i
print("ppl: ", ppl_min, ' ', p_id, data[p_id])
print("rouge: ", rougel_max, ' ', r_id, data[r_id])
print('bertscore: ', bertscore_max, ' ', b_id, data[b_id])
print("meteor: ", meteor_max, ' ', m_id, data[m_id])
print('d1: ', d1_max, ' ', d1_id, data[d1_id])
print("d2: ", d2_max, ' ', d2_id, data[d2_id])
print('bleu1: ', bleu1_max, b1_id, data[b1_id])
print('bleu4: ', bleu4_max, b4_id, data[b4_id])
print('cider: ', cider_max, c_id, data[c_id])
