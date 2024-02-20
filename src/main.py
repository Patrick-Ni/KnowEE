# /data/xfni/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/2/27 21:59
# @Author       : patrick
# @File         : main.py
# @Description  :
import argparse
import pickle
import random

import numpy as np
from tqdm import tqdm

from batch_generate import batch_emotion_generate, batch_persona_generate, batch_topic_generate, batch_wiki_generate, \
    batch_event_generate, batch_reply_generate, inside_template, outside_template, ablation_template
from load_models_and_datasets import load_model_and_tokenizer, load_generated_knowledge, load_main_datasets
import pandas as pd
from datasets import load_dataset
from utils.calc_similarity import calc_sentence_similarity
import torch

home = '~/code/KnowEE'
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model', type=str, default='flan-t5-xxl')
parser.add_argument('--emotion', action='store_true')
parser.add_argument('--wiki', action='store_true')
parser.add_argument('--persona', action='store_true')
parser.add_argument('--topic', action='store_true')
parser.add_argument('--event', action='store_true')
parser.add_argument('--ablation', action='store_true')
parser.add_argument('--ablation_name', type=str, default='emotion', choices=['emotion', 'event', 'persona', 'wiki', 'topic'])
parser.add_argument('--gen', action='store_true')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--print', action='store_true')
parser.add_argument('--check', action='store_true')
parser.add_argument('--version', type=str, default='reply')
parser.add_argument('--dataset', type=str, default='empathetic',
                    choices=['empathetic', 'dailydialog', 'odkg', 'blended_skill_talk', 'persona_chat'])
parser.add_argument('--template', type=str, default='inside', choices=['inside', 'outside'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_length', type=int, default=30)
parser.add_argument('--prompt_num', type=int, default=3)
parser.add_argument('--start', type=int, default=0)

args = parser.parse_args()


def emotion(data, model, tokenizer, device):
    go_emotions = load_dataset('go_emotions', 'raw', split='train')
    emotions = batch_emotion_generate(data, args.model, model, tokenizer, device, go_emotions,
                                      batch_size=args.batch_size, print_tag=args.print, check=args.check,
                                      file_path=home + f'results/{args.model}/{args.dataset}/checkpoint_emotion.txt')
    f = open(home + f'results/{args.model}/{args.dataset}/emotion.txt', 'a')
    for emo in emotions:
        f.write(str(emo) + '\n')


def topic(data, model, tokenizer, device):
    dailydialog = pd.read_json('../data/preprocess_data/dailydialog/train.json')
    process_dd, process_topic = [], []
    for i, utter in enumerate(dailydialog['dialogue']):
        texts = 'utterance:\n'
        for idx, u in enumerate(utter[0:4]):
            if idx % 2 == 0:
                texts += 'A: ' + u + '\n'
            else:
                texts += 'B: ' + u + '\n'
        process_dd.append(texts)
        process_topic.append(texts + 'Topic: ' + dailydialog['topic'][i].replace('_', ' ') + '\n')
    _ = batch_topic_generate(data, args.model, model, tokenizer, device, process_dd, process_topic,
                             batch_size=args.batch_size, print_tag=args.print, check=args.check,
                             file_path=home + f'results/{args.model}/{args.dataset}/topic.txt')


def persona(data, model, tokenizer, device):
    process_text, process_persona = [], []
    with open(home + 'data/prompt_datasets/process_text.txt', 'r') as f:
        for line in f.readlines():
            process_text.append(line[:-1].replace('Utterance:  ', ''))
    with open(home + 'data/prompt_datasets/process_persona.txt', 'r') as f:
        count = 0
        content = ''
        for line in f.readlines():
            if line == '\n':
                continue
            content += line
            count += 1
            if count == 2:
                process_persona.append(content)
                count = 0
                content = ''
    assert len(process_text) == len(process_persona), f'wrong process_text num'
    persona_A, persona_B = batch_persona_generate(data, args.model, model, tokenizer, device, process_text, process_persona,
                                                  print_tag=args.print, check=args.check,
                                                  batch_size=args.batch_size,
                                                  file_path=home + f'results/{args.model}/{args.dataset}/checkpoint_persona.txt')
    # prompt_num = 4 for flan-t5-xxl generating dailydialog
    assert len(persona_A) == len(persona_B), f'different number of persona A and persona B'
    with open(home + f'results/{args.model}/{args.dataset}/persona_A.txt', 'w') as f:
        for p in persona_A:
            f.write(p + '\n')
    with open(home + f'results/{args.model}/{args.dataset}/persona_B.txt', 'w') as f:
        for p in persona_B:
            f.write(p + '\n')


def wiki(data, model, tokenizer, device):
    with open('../data/plm_knowledge/preprocess_wow_train.pkl', 'rb') as f:
        wow = pickle.load(f)
    _ = batch_wiki_generate(data, args.model, model, tokenizer, device,
                            wow, print_tag=args.print, batch_size=args.batch_size, check=args.check,
                            file_path=home + f'results/{args.model}/{args.dataset}/wiki.txt')
    # prompt_num = 4 for flan-t5-xxl generating dailydialog
    # with open('../results/flan-t5/empathetic/wiki.txt', 'w') as f:
    #     for w in wikis:
    #         f.write(str(w) + '\n')


def event(data, model, tokenizer, device):
    x = pd.read_csv('/data/xfni/code/Datasets/atomic/v4_atomic_all.csv')
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
    intents, needs = batch_event_generate(data, args.model, model, tokenizer, device, intent, need,
                                          batch_size=args.batch_size, print_tag=args.print, check=args.check,
                                          file_path_need=home + f'results/{args.model}/{args.dataset}/c-need.txt',
                                          file_path_intent=home + f'results/{args.model}/{args.dataset}/c-intent.txt')
    with open(home + f'results/{args.model}/{args.dataset}/intent.txt', 'w') as f:
        for intent in intents:
            f.write(str(intent) + '\n')
    with open(home + f'results/{args.model}/{args.dataset}/need.txt', 'w') as f:
        for need in needs:
            f.write(str(need) + '\n')


def make_prompt_datasets():
    process_text, process_persona = [], []
    persona_chat = pd.read_json('../data/plm_knowledge/train_revised.json')
    for i, utter in enumerate(tqdm(persona_chat['dialogue'])):
        speaker, listener = utter[::2], utter[1::2]
        speaker_persona, listener_persona = ' '.join(persona_chat['speaker persona'][i][0:2]), ' '.join(
            persona_chat['listener persona'][i][0:2])
        simi_speaker = calc_sentence_similarity(speaker_persona, speaker)
        simi_listener = calc_sentence_similarity(listener_persona, listener)
        indices_s = torch.topk(simi_speaker, 5)[1]
        indices_l = torch.topk(simi_listener, 5)[1]
        utter_s, utter_l = 'Utterance: ', 'utterance: '
        for index in indices_s[0]:
            utter_s += ' ' + speaker[int(index)]
        for index in indices_l[0]:
            utter_l += ' ' + listener[int(index)]
        process_text.append(utter_s)
        process_text.append(utter_l)
        process_persona.append(utter_s + '\nPersona: ' + speaker_persona + '\n')
        process_persona.append(utter_l + '\nPersona: ' + listener_persona + '\n')


def set_global_seeds():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    set_global_seeds()
    data = load_main_datasets(args.dataset, home)
    data = data[args.start:200]
    tokenizer, model, device = load_model_and_tokenizer(args.model)
    # tokenizer, model, device = None, None, None

    if args.emotion:
        emotion(data, model, tokenizer, device)
    if args.persona:
        persona(data, model, tokenizer, device)
    if args.wiki:
        wiki(data, model, tokenizer, device)
    if args.topic:
        topic(data, model, tokenizer, device)
    if args.event:
        event(data, model, tokenizer, device)
    if args.gen:
        prompt_enhanced_dialogue_generation(data, model, tokenizer, device)
    if args.baseline:
        baseline_generation(data, model, tokenizer, device)
    if args.ablation:
        ablation(data, model, tokenizer, device, args.ablation_name)


def prompt_enhanced_dialogue_generation(data, model, tokenizer, device):
    _emotion, intent, need, persona_A, persona_B, _topic, _wiki = load_generated_knowledge(home + f'results/flan-t5-xxl/{args.dataset}/')
    speaker_map = {
        1: 'A',
        -1: 'B'
    }
    assert len(_emotion) == len(intent)
    assert len(_emotion) == len(need)
    assert len(_emotion) == len(_topic)
    if persona_A is not None:
        assert len(_emotion) == len(persona_A)
    if persona_B is not None:
        assert len(_emotion) == len(persona_B)
    if _wiki is not None:
        assert len(_emotion) == len(_wiki)
    utterance = []
    for idx, utters in enumerate(data):
        assert len(utters) % 2 == 1
        data = []
        speaker_tag = 1
        for jdx, u in enumerate(utters[0:-3]):
            data.append({'text': u, 'speaker': speaker_map[speaker_tag]})
            speaker_tag = -speaker_tag
        add_length = len(utters[0:-3])
        event_length = 1 if len(utters) > 3 else 0
        # if len(need[idx]) != 4:
        #         #     print(len(need[idx]))
        #         #     print(idx)
        for jdx, u in enumerate(utters[-3:]):
            try:
                data.append({'text': u, 'speaker': speaker_map[speaker_tag], 'emotion': _emotion[idx][jdx + add_length],
                             'xIntent': intent[idx][jdx + event_length].split(' | ')[0],
                             'xNeed': need[idx][jdx + event_length].split(' | ')[0]})
            except:
                print(idx)
                print(jdx + add_length)
                print(_emotion[idx][jdx + add_length])
                print(intent[idx][jdx + add_length].split(' | ')[0])
                print(need[idx][jdx + add_length].split(' | ')[0])
                import sys
                print("end")
                sys.exit()
            speaker_tag = -speaker_tag
        utterance.append(data)
    texts = []
    if args.template == 'inside':
        for idx, u in enumerate(utterance):
            p_a = persona_A[idx] if persona_A is not None else None
            p_b = persona_B[idx] if persona_B is not None else None
            w = _wiki[idx] if _wiki is not None else None
            texts.append(inside_template(u, p_a, p_b, _topic[idx], w))
    elif args.template == 'outside':
        for idx, u in enumerate(utterance):
            p_a = persona_A[idx] if persona_A is not None else None
            p_b = persona_B[idx] if persona_B is not None else None
            w = _wiki[idx] if _wiki is not None else None
            texts.append(outside_template(u, p_a, p_b, _topic[idx], w))
    else:
        raise ValueError('Wrong template type!')
    # with open(f'../results/{args.dataset}_{args.template}_200.txt', 'w') as f:
    #     for t in list(texts):
    #         f.write(t.replace('\n', '[SEP]') + '\n')
    # import sys
    # sys.exit()
    _ = batch_reply_generate(list(texts), args.model, model, tokenizer, device, batch_size=args.batch_size,
                             print_tag=args.print, check=args.check,
                             file_path=home + f'results/{args.model}/{args.dataset}/{args.version}.txt')


def baseline_generation(data, model, tokenizer, device):
    texts = []
    for utters in data:
        text = ''
        for idx, u in enumerate(utters):
            assert len(utters) % 2 == 1
            if idx % 2 == 0:
                text += 'A: ' + u + '\n'
            else:
                text += 'B: ' + u + '\n'
        text = text + 'B: '
        texts.append(text)
    _ = batch_reply_generate(list(texts), args.model, model, tokenizer, device, batch_size=args.batch_size,
                             print_tag=args.print, check=args.check,
                             file_path=home + f'results/{args.model}/{args.dataset}/{args.version}.txt')
    # top_p = 0.8 for reply 5 of flan-t5-xxl on empathetic


def ablation(data, model, tokenizer, device, ablation_name):
    _emotion, intent, need, persona_A, persona_B, _topic, _wiki = load_generated_knowledge(home + f'results/{args.model}/{args.dataset}/')
    maps = ['emotion', 'intent', 'need', 'persona_A', 'persona_B', 'topic', 'wiki']
    knowledge = [_emotion, intent, need, persona_A, persona_B, _topic, _wiki]
    if ablation_name == 'emotion' or ablation_name == 'topic' or ablation_name == 'wiki':
        idx = maps.index(ablation_name)
        knowledge.pop(idx)
        maps.pop(idx)
    elif ablation_name == 'persona':
        idx = maps.index('persona_A')
        knowledge.pop(idx)
        maps.pop(idx)
        idx = maps.index('persona_B')
        knowledge.pop(idx)
        maps.pop(idx)
    elif ablation_name == 'event':
        idx = maps.index('intent')
        knowledge.pop(idx)
        maps.pop(idx)
        idx = maps.index('need')
        knowledge.pop(idx)
        maps.pop(idx)
    else:
        raise ValueError('wrong ablation name!')
    ablation_generation(data, model, tokenizer, device, maps, knowledge, f'wo_{ablation_name}')


def ablation_generation(data, model, tokenizer, device, maps, knowledge, file_path_name):
    assert len(maps) == len(knowledge)
    speaker_map = {
        1: 'A',
        -1: 'B'
    }
    utterance = []
    for idx, utters in enumerate(data):
        assert len(utters) % 2 == 1
        data = []
        speaker_tag = 1
        for jdx, u in enumerate(utters[0:-3]):
            data.append({'text': u, 'speaker': speaker_map[speaker_tag]})
            speaker_tag = -speaker_tag
        add_length = len(utters[0:-3])
        event_length = 1 if len(utters) > 3 else 0
        # if len(need[idx]) != 4:
        #         #     print(len(need[idx]))
        #         #     print(idx)
        for jdx, u in enumerate(utters[-3:]):
            d = {'text': u, 'speaker': speaker_map[speaker_tag]}
            for kdx, k in enumerate(knowledge):
                if maps[kdx] == 'emotion':
                    d[maps[kdx]] = k[idx][jdx + add_length]
                elif maps[kdx] == 'event':
                    d['intent'] = k[idx][jdx + event_length].split(' | ')[0]
                    d['need'] = k[idx][jdx + event_length].split(' | ')[0]
            data.append(d)
            speaker_tag = -speaker_tag
        utterance.append(data)
    texts = []
    persona_A = knowledge[maps.index('persona_A')] if 'persona_A' in maps else None
    persona_B = knowledge[maps.index('persona_B')] if 'persona_B' in maps else None
    _wiki = knowledge[maps.index('wiki')] if 'wiki' in maps else None
    _topic = knowledge[maps.index('topic')] if 'topic' in maps else None
    for idx, u in enumerate(utterance):
        p_a = persona_A[idx] if persona_A is not None else None
        p_b = persona_B[idx] if persona_B is not None else None
        w = _wiki[idx] if _wiki is not None else None
        t = _topic[idx] if _topic is not None else None
        texts.append(ablation_template(u, p_a, p_b, t, w))
    _ = batch_reply_generate(list(texts), args.model, model, tokenizer, device, batch_size=args.batch_size,
                             print_tag=args.print, check=args.check,
                             file_path=home + f'results/{args.model}/{args.dataset}/{file_path_name}.txt')


if __name__ == '__main__':
    main()
