# ~/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/3/12 12:42
# @Author       : patrick
# @File         : data_wash.py
# @Description  :
import json

import pandas as pd


def del_less_3():
    for name in ['test', 'val', 'train']:
        ed = pd.read_json(f'empathetic/{name}.json')
        del_index = []
        ori_len = len(ed)
        for idx, dialog in enumerate(ed['utterance']):
            if len(dialog) < 3:
                del_index.append(idx)
        ed = ed.rename(columns={'utterance': 'dialogue'})
        ed = ed.drop(ed.index[del_index])
        print(ori_len)
        print(len(del_index))
        print(len(ed))
        for idx, dialog in enumerate(ed['dialogue']):
            if len(dialog) < 3:
                print(dialog)
        ed.to_json(path_or_buf=f'empathetic/clean_{name}.json', orient='records')


def convert_to_3():
    for name in ['test', 'val', 'train']:
        ed = pd.read_json(f'empathetic/{name}.json')
        data_s = []
        for i in range(len(ed)):
            if len(ed['dialogue'][i]) == 3:
                response = ed['response'][i]
            else:
                response = ed['dialogue'][i][3]
            data = {'situation': ed['situation'][i], 'dialogue': ed['dialogue'][i][0:3], 'label': ed['label'][i],
                    'response': response}
            data_s.append(data)
        assert len(data_s) == len(ed)
        with open(f'empathetic/{name}.json', 'w') as f:
            json.dump(data_s, f)


def convert(dataset):
    for name in ['test', 'val', 'train']:
        ds = pd.read_json(f'preprocess_data/{dataset}/{name}.json')
        data_s = []
        count = 0
        for i in range(len(ds)):
            if len(ds['dialogue'][i]) % 2 == 0:
                dialogue = ds['dialogue'][i]
            else:
                dialogue = ds['dialogue'][i][0:-1]
            if dataset == 'dailydialog':
                data = {'topic': ds['topic'][i], 'dialogue': dialogue[0:-1], 'response': dialogue[-1]}
            else:
                data = {'dialogue': dialogue[0:-1], 'response': dialogue[-1]}
            data_s.append(data)
        assert len(data_s) + count == len(ds)
        with open(f'{dataset}/{name}.json', 'w') as f:
            json.dump(data_s, f)


def get_answers(dataset):
    ed = pd.read_json(f'{dataset}/test.json')
    with open(f'{dataset}/answers.txt', 'w') as f:
        for ans in ed['response']:
            f.write(ans + '\n')


def deduplicate(model, dataset):
    emotion = []
    lack = []
    new_emotion = []
    intents = []
    persona_A = []
    persona_B = []
    topic = []
    wiki = []
    needs = []
    with open(f'../results/{model}/{dataset}/emotion.txt', 'r') as f:
        for e in f.readlines():
            emo = eval(e.replace('\n', ''))
            if len(emo) > 3:
                emo = emo[0:3]
            emotion.append(emo)
    for idx, emo in enumerate(emotion):
        if len(emo) < 3:
            lack.append(idx)
            continue
        new_emotion.append(emo)
    for ne in new_emotion:
        if len(ne) != 3:
            print("error!")
    assert len(new_emotion) + len(lack) == len(emotion)
    with open(f'../results/{model}/{dataset}/emotion.txt', 'w') as f:
        for e in emotion:
            f.write(str(e) + '\n')
    with open(f'../results/{model}/{dataset}/need.txt', 'r') as f:
        for x in f.readlines():
            need = eval(x.replace('\n', ''))
            if len(need) > 3:
                need = need[0:3]
            if len(need) < 3:
                continue
            needs.append(need)
    with open(f'../results/{model}/{dataset}/intent.txt', 'r') as f:
        for x in f.readlines():
            intent = eval(x.replace('\n', ''))
            if len(intent) > 3:
                intent = intent[0:3]
            if len(intent) < 3:
                continue
            intents.append(intent)
    with open(f'../results/{model}/{dataset}/persona_A.txt', 'r') as f:
        for x in f.readlines():
            persona_A.append(x.replace('\n', ''))
    with open(f'../results/{model}/{dataset}/persona_B.txt', 'r') as f:
        for x in f.readlines():
            persona_B.append(x.replace('\n', ''))
    with open(f'../results/{model}/{dataset}/topic.txt', 'r') as f:
        for x in f.readlines():
            topic.append(x.replace('\n', ''))
    with open(f'../results/{model}/{dataset}/wiki.txt', 'r') as f:
        for x in f.readlines():
            wiki.append(x.replace('\n', ''))
    persona_A = delete_target_index(persona_A, lack)
    persona_B = delete_target_index(persona_B, lack)
    topic = delete_target_index(topic, lack)
    wiki = delete_target_index(wiki, lack)
    for i in range(10):
        print(intents[i])
        print(new_emotion[i])
        print(topic[i])
        print(wiki[i])
        print(persona_A[i])
        print(persona_B[i])
    assert len(new_emotion) == len(intents)
    assert len(intents) == len(wiki)
    with open(f'../results/{model}/{dataset}/emotion.txt', 'w') as f:
        for e in new_emotion:
            f.write(str(e) + '\n')
    with open(f'../results/{model}/{dataset}/intent.txt', 'w') as f:
        for e in intents:
            f.write(str(e) + '\n')
    with open(f'../results/{model}/{dataset}/persona_A.txt', 'w') as f:
        for e in persona_A:
            f.write(e + '\n')
    with open(f'../results/{model}/{dataset}/persona_B.txt', 'w') as f:
        for e in persona_B:
            f.write(e + '\n')
    with open(f'../results/{model}/{dataset}/topic.txt', 'w') as f:
        for e in topic:
            f.write(e + '\n')
    with open(f'../results/{model}/{dataset}/wiki.txt', 'w') as f:
        for e in wiki:
            f.write(e + '\n')
    with open(f'../results/{model}/{dataset}/need.txt', 'w') as f:
        for e in needs:
            f.write(str(e) + '\n')


def delete_target_index(list_given, index_to_delete):
    for index in reversed(index_to_delete):
        list_given.pop(index)
    return list_given


if __name__ == '__main__':
    # get_answers('blended_skill_talk')

    # deduplicate('flan-t5-xxl', 'empathetic')
    # deduplicate('flan-t5-xxl', 'dailydialog')
    # deduplicate('flan-t5-xxl', 'blended_skill_talk')
    convert('persona_chat')
    get_answers('persona_chat')
