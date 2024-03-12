# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/12/20 10:28
# @Author : Patrick Ni
# @File : combine_files.py
# @description:

#  test: emotion, intent, wiki
# train: emotion, intent, wiki, topic
# val: emotion, intent, wiki
import pandas as pd

data_train = pd.read_json('/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/train.json')
data_test = pd.read_json('/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/test.json')
data_val = pd.read_json('/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/val.json')

data_train.rename(columns={'knowledge': "wiki"}, inplace=True)
data_test.rename(columns={'knowledge': "wiki"}, inplace=True)
data_val.rename(columns={'knowledge': "wiki"}, inplace=True)

data_train.to_json(
    path_or_buf='/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/train.json',
    orient='records')
data_test.to_json(
    path_or_buf='/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/test.json',
    orient='records')
data_val.to_json(
    path_or_buf='/data/xfni/code/PMDialogueSystem/data/preprocess_data/open_dialog_kg/val.json',
    orient='records')
