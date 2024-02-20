# ~/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/2/14 16:14
# @Author       : patrick
# @File         : interactive.py
# @Description  : an interactive program for chat bot
import argparse
import pickle
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert
from datasets import load_dataset
import random
import torch

ST = SBert("all-MiniLM-L12-v2")


def calc_sentence_similarity(sentence_list_1, sentence_list_2):
    # multi-sentences: n-list1, m-list2, return tensor with shape(nxm)
    # single-sentences: return tensor with shape 1x1
    embeddings1 = ST.encode(sentence_list_1)
    embeddings2 = ST.encode(sentence_list_2)
    return cos_sim(embeddings1, embeddings2)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='flan-t5-xxl')

parser.add_argument('--do_sample', action='store_true')
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--temperature', type=int, default=0.9)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument('--return_num', type=int, default=5)

# greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False.
# multinomial sampling by calling sample() if num_beams=1 and do_sample=True.
# beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False.
# beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True.
# diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1.
args = parser.parse_args()


def load_model_and_tokenizer():
    max_memory_mapping = {0: "0GB", 1: "15GB", 2: "15GB", 3: "15GB", 4: "15GB", 5: "12GB", 6: "12GB", 7: "6GB"}
    device = 'cuda:7'
    if 'flan-t5' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{args.model}')
        # model = AutoModelForSeq2SeqLM.from_pretrained('~/code/Pretrained_Models/flan-t5-xxl', device_map="auto",
        # max_memory=max_memory_mapping,offload_folder="offload", offload_state_dict=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(f'~/code/Pretrained_Models/{args.model}', device_map="auto",
                                                      max_memory=max_memory_mapping)

    elif 'gpt-neox' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{args.model}')
        # if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.encode('[PAD]')[0]
        model = AutoModelForCausalLM.from_pretrained(f'~/code/Pretrained_Models/{args.model}', device_map="auto",
                                                     max_memory=max_memory_mapping)
    elif 'opt' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(f'~/code/Pretrained_Models/{args.model}')
        model = AutoModelForCausalLM.from_pretrained(f'~/code/Pretrained_Models/{args.model}', device_map="auto",
                                                     max_memory=max_memory_mapping)
    else:
        raise ValueError(f'wrong model type {args.model_name}, should be one of flan-t5, opt, gpt-neox series.')
    return tokenizer, model, device


def batch_generate(texts,
                   model,
                   tokenizer,
                   device,
                   max_length=50,
                   do_sample=True,
                   num_beams=1,
                   top_k=0,
                   top_p=0.9,
                   temperature=0.9):
    if isinstance(texts, list):
        _input = tokenizer(texts, max_length=(len(texts) // 100 + 1) * 100, return_tensors="pt", truncation=True,
                           padding=True, add_special_tokens=True).to(device)
    else:
        _input = tokenizer(texts, return_tensors="pt", add_special_tokens=True).to(device)

    model.eval()

    # torch.cuda.empty_cache()
    try:
        results = model.generate(_input["input_ids"], attention_mask=_input['attention_mask'], max_length=max_length,
                                 num_beams=num_beams, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
        results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except RuntimeError:
        raise RuntimeError('CUDA OUT OF MEMORY WHEN GENERATING RESPONSE!')
    return results
    tag = 0
    for utter in utterance:
        if 'emotion' in utter.keys():
            continue
        utter['emotion'] = results[tag]
        tag = tag + 1

    for utter in utterance:
        if 'wiki' in utter.keys():
            continue
        utter['wiki'] = results[tag]
        tag = tag + 1

    for utter in utterance:
        if 'xIntent' in utter.keys():
            continue
        utter['xIntent'] = results[tag]
        tag = tag + 1
    for utter in utterance:
        if 'xNeed' in utter.keys():
            continue
        utter['xNeed'] = results[tag]
        tag = tag + 1
    if len(utterance) == 1:
        return results[-2], None, results[-1]
    else:
        return results[-3], results[-2], results[-1]


def emotion_input_generate(utterance,
                           model,
                           tokenizer,
                           device,
                           prompt_dataset,
                           prompt_num=8
                           ):
    # utterance: [{text: utter_1, speaker:A, emotion: xxx},{text: utter_2, speaker:B, emotion: xxx} ]
    # emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    #                   'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    #                   'optimism',
    #                   'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    # use GoEmotions dataset as prompt dataset
    prompt_text = prompt_dataset['text']
    sample_ids = random.sample(range(len(prompt_text)), 50000)
    sample_texts = [prompt_text[i] for i in sample_ids]
    utter_emotion_query = []
    for utter in utterance:
        if 'emotion' in utter.keys():
            continue
        simi = calc_sentence_similarity(utter['text'], sample_texts)
        values, indices = torch.topk(simi, 803)
        prompt = ''
        count = prompt_num
        for index in indices[0]:
            _dict = dict(prompt_dataset[sample_ids[int(index)]])
            _dict.pop('rater_id')
            _dict.pop('created_utc')
            emotion = [k for k, v in _dict.items() if v == 1]
            if emotion:
                prompt += _dict['text'] + '\nThe emotion of this text is: ' + emotion[0] + '\n'
                count = count - 1
                if count == 0:
                    break
        utter_emotion_query.append(prompt + utter['text'] + '\nThe emotion of this text is: ')
    emotion_results = batch_generate(utter_emotion_query, model,
                                     tokenizer,
                                     device,
                                     max_length=2)
    tag = 0
    for utter in utterance:
        if 'emotion' in utter.keys():
            continue
        utter['emotion'] = emotion_results[tag]
        tag = tag + 1
    return utter_emotion_query


def topic_query_generate(utterance,
                         model,
                         tokenizer,
                         device,
                         process_dd,
                         process_topic,
                         prompt_num=5):
    # data: dataframe: [{dialogue: ['utter1','utter2','utter3',....],response:'...'},{},{},]
    # max_memory_mapping = {'cpu': "10GB", 0: "0GB", 1: "0GB", 2: "0GB", 3: "0GB", 4: "0GB", 5: "0GB", 6: "0GB", 7: "0GB"}
    assert len(process_dd) == len(process_topic), f'Wrong num: {len(process_dd)} and {len(process_topic)}'
    text = 'utterance: '
    for utter in utterance:
        text += utter['text'] + ' '
    simi = calc_sentence_similarity(text, process_dd)
    indices = torch.topk(simi, 10)[1]

    prompt = ''
    count = prompt_num
    for index in indices[0]:
        prompt = prompt + process_topic[int(index)]
        count -= 1
        if count == 0:
            break
    topic_result = batch_generate(prompt + text + '\nThe topic of this utterance is: ', model, tokenizer, device,
                                  max_length=3)[0]
    return topic_result


def event_query_generate(utterance,
                         model,
                         tokenizer,
                         device,
                         intent,
                         need,
                         prompt_num=8,
                         ):
    answer_template = {
        'xNeed': 'speaker needed {}',
        'xIntent': 'speaker wanted {}'
    }

    def sub_function(text, simi, process_texts, process_relations, tag='xIntent', steer='intent: '):
        prompt = ''
        count = prompt_num
        indices = torch.topk(simi, 500)[1]
        for index in indices[0]:
            prompt_answers = trans_str(process_relations[int(index)])
            if not prompt_answers:
                continue
            prompt += process_texts[int(index)] + '\n' + steer + answer_template[tag].format(
                ' '.join(prompt_answers[0:5])) + '\n'
            count -= 1
            if count == 0:
                break
        _input = prompt + text + '\n' + steer
        return _input

    utter_intent_query = []
    utter_need_query = []
    for utter in utterance:
        if 'event' in utter.keys():
            continue
        simi_intent = calc_sentence_similarity(utter['text'], list(intent['event']))
        simi_need = calc_sentence_similarity(utter['text'], list(need['event']))
        # utter['xIntent'] = sub_function(utter['text'], simi_intent, intent['event'], intent['xIntent'], tag='xIntent',
        #                                 steer='intent: ')
        # utter['xNeed'] = sub_function(utter['text'], simi_need, need['event'], need['xNeed'], tag='xNeed', steer='need: ')
        utter_intent_query.append(sub_function(utter['text'], simi_intent, intent['event'], intent['xIntent'], tag='xIntent',
                                               steer='intent: '))
        utter_need_query.append(
            sub_function(utter['text'], simi_need, need['event'], need['xNeed'], tag='xNeed', steer='need: '))
    intent_result = batch_generate(utter_intent_query, model, tokenizer, device, max_length=50)
    need_result = batch_generate(utter_need_query, model, tokenizer, device, max_length=50)
    tag = 0
    for utter in utterance:
        if 'xIntent' in utter.keys():
            continue
        utter['xIntent'] = intent_result[tag]
        utter['xNeed'] = need_result[tag]
        tag = tag + 1


def persona_query_generate(utterance,
                           model,
                           tokenizer,
                           device,
                           process_text,
                           process_persona,
                           prompt_num=5,
                           ):
    # use persona chat as prompt dataset

    speaker_A, speaker_B = 'utterance: ', 'utterance: '
    for utter in utterance:
        if utter['speaker'] == 'A':
            speaker_A += utter['text'] + ' '
        else:
            speaker_B += utter['text'] + ' '
    simi_A = calc_sentence_similarity(speaker_A, process_text)
    simi_B = calc_sentence_similarity(speaker_B, process_text) if speaker_B != 'utterance: ' else None
    indices_A = torch.topk(simi_A, 500)[1]
    indices_B = torch.topk(simi_B, 500)[1] if simi_B is not None else simi_B

    def persona_model_output(indices, speaker_text):
        prompt = ''
        count = prompt_num
        for index in indices[0]:
            prompt = prompt + process_persona[int(index)]
            count -= 1
            if count == 0:
                break
        _input = prompt + speaker_text + '\nThe persona of this speaker is: '
        return _input

    persona_query = [persona_model_output(indices_A, speaker_A)]

    if indices_B is not None:
        persona_query.append(persona_model_output(indices_B, speaker_B))
    persona = batch_generate(persona_query, model, tokenizer, device, max_length=50)
    persona_A = persona[0]
    if len(persona) == 2:
        persona_B = persona[1]
    else:
        persona_B = None
    return persona_A, persona_B


def wiki_query_generate(utterance,
                        model,
                        tokenizer,
                        device,
                        prompt_dataset,
                        prompt_num=5
                        ):
    # use Wizard of Wikipedia as prompt dataset
    utter_wiki_query = []
    prompt_text = prompt_dataset['text']
    sample_ids = random.sample(range(len(prompt_text)), 50000)
    sample_texts = [prompt_text[i] for i in sample_ids]
    for utter in utterance:
        if 'wiki' in utter.keys():
            continue
        simi = calc_sentence_similarity(utter['text'], sample_texts)
        values, indices = torch.topk(simi, 50)
        prompt = ''
        count = prompt_num
        for index in indices[0]:
            prompt += prompt_dataset['text'][int(index)] + '\nThe wikipedia knowledge of this text is: ' + \
                      prompt_dataset['knowledge'][
                          int(index)] + '\n'
            count = count - 1
            if count == 0:
                break
        utter_wiki_query.append(prompt + utter['text'] + '\nThe wikipedia knowledge of this text is: ')
    wiki_result = batch_generate(utter_wiki_query, model,
                                 tokenizer,
                                 device, max_length=50)
    tag = 0
    for utter in utterance:
        if 'wiki' in utter.keys():
            continue
        utter['wiki'] = wiki_result[tag]
        tag = tag + 1


def load_prompt_datasets():
    # prompt_file_path = args.data_root_path + 'plm_knowledge/preprocess_dd_train.json'
    persona_chat = pd.read_json('../../PMDialogueSystem/data/plm_knowledge/train_revised.json')
    dailydialog = pd.read_json('../../PMDialogueSystem/data/plm_knowledge/preprocess_dd_train.json')
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
        process_persona.append(speaker + '\nThe persona of this speaker is: ' + s_persona + '\n')
        process_persona.append(listener + '\nThe persona of this speaker is: ' + l_persona + '\n')
    for i, utter in enumerate(dailydialog['dialogue']):
        process_dd.append('utterance: ' + ' '.join(utter))
        process_topic.append(
            'utterance: ' + ' '.join(utter) + '\nThe topic of this utterance is: ' + dailydialog['topic'][i].replace('_',
                                                                                                                     '') + '\n')
    with open('../data/plm_knowledge/preprocess_wow_train.pkl', 'rb') as f:
        wow = pickle.load(f)
    GoEmotions = load_dataset('go_emotions', 'raw')
    x = pd.read_csv('../../Datasets/atomic/v4_atomic_all.csv')
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


def show_middle_string(string, length, completion='-'):
    assert length >= len(string)
    left = length - len(string)
    show_str = string.rjust(left // 2 + len(string), completion)
    show_str = show_str.ljust(length, completion)
    print(show_str)


def extract_events_from_text(sentence, extractor):
    eventualities, _ = extractor.extract_from_text(sentence, in_order=True)
    event = sum(eventualities, [])
    x = str(event)[1:-1].split(', ')
    return x


def trans_str(string):
    string = string.replace('["none"]', '')
    if 'none' in string:
        print("你是在玩我？？？？")
    strings = string.split(']')
    lists = []
    for s in strings:
        if len(s) <= 1 or s[0] != '[':
            continue
        s = eval(s + ']')
        lists.extend(s)
    return lists


def outside_template(utterance, persona_A, persona_B, topic, wiki=None):
    emotion_A = utterance[-1]['emotion']
    intent_A = utterance[-1]['xIntent']
    need_A = utterance[-1]['xNeed']
    if len(utterance) == 1:
        supply = False
        emotion_B = intent_B = need_B = None
    else:
        emotion_B = utterance[-2]['emotion']
        intent_B = utterance[-2]['xIntent']
        need_B = utterance[-2]['xNeed']
        supply = True

    # background_knowledge = "Please generate a response based on knowledge below, including persona knowledge, emotion knowledge, topic knowledge, \
    # intent and need knowledge ( describes what do speakers want to do ). Please keep the response diverse.\
    # \nBackground Knowledge: Persona of A: " + persona_A
    background_knowledge = "Please generate a response based on wikipedia knowledge below. Please keep the response diverse.\nBackground Knowledge: Topic: " + \
                           topic
    background_knowledge = background_knowledge + "[SEP]Persona of A: " + persona_A if persona_A is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Persona of B: ' + persona_B if (
            supply and persona_B is not None) else background_knowledge
    background_knowledge += '[SEP]Emotion of A: ' + emotion_A
    background_knowledge = background_knowledge + '[SEP]Emotion of B: ' + emotion_B if supply else background_knowledge
    background_knowledge += '[SEP]Intent of A: ' + intent_A + '[SEP]Need of A: ' + need_A
    background_knowledge = background_knowledge + '[SEP]Intent of B: ' + intent_B + '[SEP]Need of B: ' + need_B if supply \
        else background_knowledge
    if wiki is not None:
        background_knowledge += '[SEP]Wikipedia Knowledge: ' + wiki
    background_knowledge += '\nDialogue:\n'
    text = tag = ''
    for utter in utterance:
        text += utter['speaker'] + ": " + utter['text'] + '\n'
        tag = utter['speaker']
    text = background_knowledge + text
    assert tag == 'A'
    text = text + 'B: '
    return text


def get_next_utterance(utterance,
                       model,
                       tokenizer,
                       device,
                       max_length=500,
                       return_num=1,
                       print_tag=1,
                       do_sample=True,
                       num_beams=1,
                       top_k=0,
                       top_p=0.9,
                       temperature=0.9, ):
    if print_tag == 1:
        print(utterance)
    _input = tokenizer(utterance, max_length=(len(utterance) // 100 + 1) * 100, return_tensors="pt", truncation=True,
                       padding=True, add_special_tokens=True).to(device)

    model.eval()

    # torch.cuda.empty_cache()
    try:
        results = model.generate(_input["input_ids"], attention_mask=_input['attention_mask'], max_length=max_length,
                                 num_beams=num_beams, num_return_sequences=return_num, do_sample=do_sample, top_k=top_k,
                                 top_p=top_p,
                                 temperature=temperature)
        results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except RuntimeError:
        raise RuntimeError('CUDA OUT OF MEMORY WHEN GENERATING RESPONSE!')
    if print_tag == 1:
        print(results)
    return results[0]


def main():
    # utterance = [{"text": "how long will it take us to drive to London ?", 'speaker': 'A'},
    #              {
    #                  'text': "I think it ' s a distance of 180 kilometers from here to London , so it should be a two-hour drive on the motorway .",
    #                  'speaker': 'B'},
    #              {"text": "that ' s unless there is a traffic jam . It could take three hours .", 'speaker': 'A'},
    #              {"text": "you ' re right . We will be able to travel at high speeds at the beginning and end of the journey , \
    #              because we will be in built-up areas .", 'speaker': 'B'},
    #              {"text": "so , shall we allow three hours to cover the distance ?", 'speaker': 'A'},
    #              {"text": "no . let me take a look ... it ' s longer than my car .", 'speaker': 'B'},
    #              {"text": "ok . You haven ' t seen my company car , have you ?", 'speaker': 'A'},
    #              {'text': "I think it ' s over five meters long . I can ' t remember exactly . \
    #              It has a maximum speed of over 200 kilometers an hour .", 'speaker': 'B'}]
    utterance = []
    show_middle_string('Loading Model', 50)
    tokenizer, model, device = load_model_and_tokenizer()
    print(f'Model: {args.model}')
    print(f'Device: {device}')
    show_middle_string('Loading Prompt Datasets', 50)
    GoEmotions, wow, process_text, process_persona, process_dd, process_topic, intent, need = load_prompt_datasets()
    print(f'Emotion: GoEmotions')
    print(f'Wikipedia: Wizard of Wikipedia')
    show_middle_string("Welcome!", 50)
    while True:
        content = input("User：")
        if content == "bye":
            show_middle_string("End of Session!", 50)
            break
        utterance.append({'text': content, 'speaker': 'A'})
        show_middle_string('Getting Knowledge', 50)
        emotion_input_generate(utterance, model, tokenizer, device, GoEmotions)
        wiki_query_generate(utterance, model, tokenizer, device, wow)
        event_query_generate(utterance, model, tokenizer, device, intent, need)
        persona_A, persona_B = persona_query_generate(utterance, model, tokenizer, device, process_text, process_persona)
        topic = topic_query_generate(utterance, model, tokenizer, device, process_dd, process_topic)
        print(utterance)
        print(persona_A)
        print(persona_B)
        print(topic)
        context = outside_template(utterance, persona_A, persona_B, topic, wiki=utterance[-1]['wiki'])
        replay = get_next_utterance(context, model, tokenizer, device)
        print('Bot: ' + replay)
        utterance.append({'text': replay, 'speaker': 'B'})


def test():
    t_utter = [{'text': 'Hello! What is your name?', 'speaker': 'A', 'emotion': 'neutral',
                'wiki': ['Hello is a greeting composed of a greeting, name, and eye contact'],
                'xIntent': ['speaker wants to know your name'],
                'xNeed': ['ask for name']}]
    pa = ['I am looking for friends.', 'i have a crush on you.']
    pb = None
    t = 'relationship'
    r = outside_template(t_utter, pa, pb, t)
    to, mo, de = load_model_and_tokenizer()
    print(get_next_utterance(r, mo, to, de))


if __name__ == '__main__':
    main()
