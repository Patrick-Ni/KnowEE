# ~/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/2/28 20:54
# @Author       : patrick
# @File         : load_models_and_datasets.py
# @Description  :
import json

import pandas as pd
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel


def load_prompt_datasets():
    # prompt_file_path = args.data_root_path + 'plm_knowledge/preprocess_dd_train.json'
    persona_chat = pd.read_json('../data/plm_knowledge/train_revised.json')
    dailydialog = pd.read_json('../data/preprocess_data/dailydialog/train.json')
    process_text, process_persona = [], []
    process_dd, process_topic = [], []
    for i, utter in enumerate(persona_chat['dialogue']):
        speaker, listener = 'utterance: ', 'utterance: '
        for idx, x in enumerate(utter):
            if idx % 2 == 0:
                speaker = speaker + x + ' '
            else:
                listener = listener + x + ' '
        s_persona = ' '.join(persona_chat['speaker persona'][i])
        l_persona = ' '.join(persona_chat['listener persona'][i])
        process_text.append(speaker)
        process_text.append(listener)
        process_persona.append(speaker + '\nPersona: ' + s_persona + '\n')
        process_persona.append(listener + '\nPersona: ' + l_persona + '\n')
    for i, utter in enumerate(dailydialog['dialogue']):
        process_dd.append('utterance: ' + ' '.join(utter))
        process_topic.append(
            'utterance: ' + ' '.join(utter) + '\nTopic: ' + dailydialog['topic'][i].replace('_', ' ') + '\n')
    with open('../data/plm_knowledge/preprocess_wow_train.pkl', 'rb') as f:
        wow = pickle.load(f)
    GoEmotions = load_dataset('go_emotions', 'raw')
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
    # intent.drop(intent[(intent.xIntent == '["none"]') | (intent.xIntent == '["none"]["none"]') | (intent.xIntent == '["none"]["none"]["none"]') | (
    #         intent.xIntent == '["none"]["none"]["none"]["none"]') | (intent.xIntent == '["none"]["none"]["none"]["none"]["none"]')].index,
    #             inplace=True)
    # need.drop(need[(need.xNeed == '["none"]') | (need.xNeed == '["none"]["none"]') | (need.xNeed == '["none"]["none"]["none"]') | (
    #         need.xNeed == '["none"]["none"]["none"]["none"]') | (need.xNeed == '["none"]["none"]["none"]["none"]["none"]')].index,
    #           inplace=True)
    return GoEmotions['train'], wow, process_text, process_persona, process_dd, process_topic, intent, need


def load_main_datasets(dataset, root):
    if dataset == 'empathetic':
        data = pd.read_json(root + 'data/empathetic/test.json')
    elif dataset == 'dailydialog':
        data = pd.read_json(root + 'data/dailydialog/test.json')
    elif dataset == 'odkg':
        data = pd.read_json(root + 'data/open_dialog_kg/test.json')
    elif dataset == 'blended_skill_talk':
        data = pd.read_json(root + 'data/blended_skill_talk/test.json')
    elif dataset == 'persona_chat':
        data = pd.read_json(root + 'data/persona_chat/test.json')
    else:
        raise ValueError('Wrong dataset name: {dataset}, expected one of empathetic, dailydialog, odkg, blended_skill_talk, persona_chat')
    return list(data['dialogue'])  # [[utter1,utter2,...],[utter1,utter2]]


def load_model_and_tokenizer(model_name):
    if 'flan-t5' in model_name:
        device = 'cuda:0'
        max_memory_mapping = {0: "0GB", 1: "10GB", 2: "13GB", 3: "16GB", 4: "16GB", 5: "0GB", 6: "0GB", 7: "24GB", 'cpu': '20GB'}
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{model_name}')
        # '\n': 3
        # model = AutoModelForSeq2SeqLM.from_pretrained('~/code/Pretrained_Models/flan-t5-xxl', device_map="auto",
        # max_memory=max_memory_mapping,offload_folder="offload", offload_state_dict=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(f'~/code/Pretrained_Models/{model_name}', device_map="auto",
                                                      max_memory=max_memory_mapping)

    elif 'gpt-neox' in model_name:
        device = 'cuda:0'
        max_memory_mapping = {0: "10GB", 1: "20GB", 2: "20GB", 3: "20GB", 4: "20GB", 5: "20GB", 6: "20GB", 7: "0GB",
                              'cpu': '0GB'}
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{model_name}', model_max_length=2048)
        # '\n': 187
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(f'~/code/Pretrained_Models/{model_name}', device_map="auto",
                                                     max_memory=max_memory_mapping)
    elif 'opt' in model_name:
        device = 'cuda:0'
        max_memory_mapping = {0: "0GB", 1: "10GB", 2: "13GB", 3: "16GB", 4: "16GB", 5: "0GB", 6: "0GB", 7: "24GB",
                              'cpu': '20GB'}
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{model_name}', model_max_length=2048, padding_side='left')
        # '\n': 50118
        model = AutoModelForCausalLM.from_pretrained(f'~/code/Pretrained_Models/{model_name}', device_map="auto",
                                                     max_memory=max_memory_mapping)
    elif 'chat' in model_name:
        device = 'cuda:1'
        tokenizer = AutoTokenizer.from_pretrained("~/code/Pretrained_Models/chatglm-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("~/code/Pretrained_Models/chatglm-6b", trust_remote_code=True).half().to(device)
    else:
        raise ValueError(f'wrong model type {model_name}, should be one of flan-t5, opt, gpt-neox series.')
    return tokenizer, model, device


def deduplicate_datasets():
    for name in ['train', 'test', 'val']:
        dailydialog = pd.read_json(f'../data/preprocess_data/dailydialog/{name}.json')
        print(len(dailydialog))
        new = []
        for i in range(len(dailydialog)):
            tag = 0
            new_data = {'dialogue': dailydialog.iloc[i]['dialogue'], 'topic': dailydialog.iloc[i]['topic']}
            for news in new:
                if news['dialogue'] == new_data['dialogue']:
                    tag = 1
                    break
            if tag == 0:
                new.append(new_data)
        print(len(new))
        with open(f'../data/preprocess_data/dailydialog/new_{name}.json', 'w') as f:
            json.dump(new, f)


def load_generated_knowledge(path):
    emotion, intent, need, persona_A, persona_B, topic, wiki = [], [], [], [], [], [], []
    with open(path + 'emotion.txt', 'r') as f:
        for line in f.readlines():
            emotion.append(eval(line.replace('\n', '')))
    with open(path + 'intent.txt', 'r') as f:
        for line in f.readlines():
            intent.append(eval(line.replace('\n', '')))
    with open(path + 'need.txt', 'r') as f:
        for line in f.readlines():
            need.append(eval(line.replace('\n', '')))
    try:
        with open(path + 'persona_A.txt', 'r') as f:
            for line in f.readlines():
                persona_A.append(line.replace('\n', ''))
    except FileNotFoundError:
        persona_A = None
    try:
        with open(path + 'persona_B.txt', 'r') as f:
            for line in f.readlines():
                persona_B.append(line.replace('\n', ''))
    except FileNotFoundError:
        persona_B = None
    with open(path + 'topic.txt', 'r') as f:
        for line in f.readlines():
            topic.append(line.replace('\n', ''))
    try:
        with open(path + 'wiki.txt', 'r') as f:
            for line in f.readlines():
                wiki.append(line.replace('\n', ''))
    except FileNotFoundError:
        wiki = None
    return emotion, intent, need, persona_A, persona_B, topic, wiki


def get_answers():
    data = pd.read_json('~/code/CurrentWork/data/preprocess_data/blended_skill_talk/test.json')
    data = list(data['dialogue'])
    with open('../eval/bst_answers.txt', 'w') as f:
        for d in data:
            f.write(d[-1] + '\n')


if __name__ == '__main__':
    get_answers()
