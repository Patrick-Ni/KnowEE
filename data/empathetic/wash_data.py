# /user/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Patrick
# @Time: 2022/10/12 21:35
# @File: wash_data.py
# description:
import pandas as pd
import json

with open('parsed_emotion_Ekman_intent_train.json', 'r') as f:
    data = json.load(f)


def clean(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    return sentence


word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
              "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
              "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
              "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not", "weren't": "were not", "won't": "will not",
              "everyone's": "everyone is", "i'll": "i will", "<fist bump>": "", "haven't": "have not", "!!": "!", "..": ".", " --": ",", "--": ", ",
              "she's": "she is", "he's": "he is", "we're": "we are", " -": "", "wouldn't": "would not", "they'll": "they will", "she'll": "she will",
              "he'll": "he will", "why's": "why is", "alllll": "all", "how's": "how is", "idk": "i do not know", "it'll": "it will", " :'(": ".",
              " lol": "",
              "there's": "there is", "there're": "there are", " :)": "", " :(": "", " :-(": "", ">": "", "<": "", "shouldn't": "should not",
              "soooo": "so", "arghhhhhhhhh": "argh",
              " ;-)": ""}

clean_saves = []
for x in data:
    situation = clean(x['prompt'])  # str
    temp_u = x['utterence']  # list
    utterance = []
    label = x['label']
    emotion = x['emotions']  # str
    intent = x['intents']  # list
    for utter in temp_u[0:len(temp_u) - 1]:
        if "|||| NEW CONVERSATION |||" in utter:
            continue
        if "He'" == utter:
            continue
        while utter != clean(utter):
            utter = clean(utter)
        utterance.append(clean(utter))
        if "'" in utter:
            print(utter)
    response = clean(temp_u[-1])
    utterance_intents = x['intents'][0:len(x['intents']) - 1]
    response_intents = x['intents'][-1]
    dicts = {'situation': situation, 'label': label,
             "utterance": utterance, "response": response, 'utterance_emotion': emotion[0:len(emotion) - 1], "response_emotion": emotion[-1],
             "utterance_intents": utterance_intents,
             "response_intents": response_intents}
    clean_saves.append(dicts)
with open('clean_train.json', 'w') as f:
    json.dump(clean_saves, f)
