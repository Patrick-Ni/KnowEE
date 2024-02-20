import sys
import random
import torch
from tqdm import tqdm

from utils.calc_similarity import calc_embeddings
from sentence_transformers.util import cos_sim


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def cut_list(lists, cut_len):
    """
    将列表拆分为指定长度的多个列表
    :param lists: 初始列表
    :param cut_len: [每个列表的长度]
    :return: 一个二维数组 [[x,x],[x,x]]
    """
    res_data = []
    assert len(lists) == sum(cut_len), f'Wrong lists length:{len(lists)} is not equal to sum of cut length: {sum(cut_len)}'
    start = 0
    for cut in cut_len:
        cut_a = lists[start:start + cut]
        start = start + cut
        res_data.append(cut_a)

    return res_data


def check_token_len(tokenizer, texts):
    max_len = 0
    sum_len = 0
    over_num = 0
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
        over_num = over_num + 1 if token_len == 512 else over_num
    return max_len, over_num / count, sum_len / count, count


def batch_emotion_generate(utterances,
                           model_name,
                           model,
                           tokenizer,
                           device,
                           prompt_dataset,
                           batch_size=16,
                           prompt_num=6,
                           print_tag=False,
                           check=False,
                           return_num=1,
                           do_sample=False,
                           num_beams=1,
                           top_k=0,
                           top_p=1,
                           temperature=1,
                           file_path=None
                           ):
    # utterance: [[text1,text2,text3... ],[text1,....]]
    # emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance',
    # 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    #                   'disapproval', 'disgust', 'embarrassment', 'excitement',
    #                   'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    #                   'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    # use GoEmotions dataset as prompt dataset
    assert type(utterances[0]) == list, f'Wrong Utterance Elements Type: {type(utterances[0])}, expected list!'
    prompt_text = prompt_dataset['text']
    sample_ids = random.sample(range(len(prompt_text)), 50000)
    sample_texts = [prompt_text[i] for i in sample_ids]
    sample_embeddings = calc_embeddings(sample_texts)

    index, texts, emotions, input_texts = [], [], [], []
    for utterance in utterances:
        index.append(len(utterance))
        for utter in utterance:
            texts.append(utter)
    batch_texts = split_batch(texts, 256)
    for batch in tqdm(batch_texts):
        simi = cos_sim(calc_embeddings(batch), sample_embeddings)
        values, indices = torch.topk(simi, 803)
        for main_index in range(len(batch)):
            prompt = ''
            count = prompt_num
            differ = []
            for sub_index in indices[main_index]:
                _dict = dict(prompt_dataset[sample_ids[int(sub_index)]])
                _dict.pop('rater_id')
                _dict.pop('created_utc')
                emotion = [k for k, v in _dict.items() if v == 1]
                if emotion:
                    if _dict['text'] in differ:
                        continue
                    prompt += _dict['text'] + '\nEmotion: ' + emotion[0] + '\n'
                    differ.append(_dict['text'])
                    count = count - 1
                    if count == 0:
                        break
            input_texts.append(
                'Please output the emotion of this text. Please give the answer in English.\n' + prompt + batch[
                    main_index] + '\nEmotion: ')
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, input_texts)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    batch_input = split_batch(input_texts, batch_size)

    for batch in tqdm(batch_input):
        tokenize_inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(
            device)
        if print_tag:
            print(batch[0])
            print(tokenize_inputs['input_ids'].shape)
        model.eval()

        # torch.cuda.empty_cache()
        if 'flan-t5' in model_name:
            results = model.generate(tokenize_inputs["input_ids"], attention_mask=tokenize_inputs['attention_mask'],
                                     max_length=4, num_beams=num_beams, num_return_sequences=return_num,
                                     do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
            results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        elif 'gpt-neox' in model_name:
            max_token_length = int(tokenize_inputs["input_ids"].shape[-1])
            results = model.generate(tokenize_inputs["input_ids"], pad_token_id=tokenizer.eos_token_id,
                                     attention_mask=tokenize_inputs['attention_mask'],
                                     max_length=max_token_length + 50, num_beams=num_beams,
                                     num_return_sequences=return_num, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                     temperature=temperature)
            results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for r in results:
                print(r)
            results[0] = results[0].replace(tokenize_inputs, '')
        elif 'chatglm' in model_name:
            assert batch_size == 1
            results, history = model.chat(tokenizer, batch[0], history=[])
            results = [str(results.encode('utf-8'))]
        else:
            results = []
        if print_tag:
            print(results)
            print_tag = False
        # torch.cuda.empty_cache()
        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write(str(results) + '\n')
        emotions.extend(results)
    emotions = cut_list(emotions, index)
    return emotions


def batch_topic_generate(utterances,
                         model_name,
                         model,
                         tokenizer,
                         device,
                         process_dd,
                         process_topic,
                         batch_size=12,
                         prompt_num=3,
                         max_length=30,
                         return_num=1,
                         print_tag=False,
                         check=False,
                         do_sample=True,
                         num_beams=1,
                         top_k=0,
                         top_p=0.9,
                         temperature=0.9,
                         file_path=None
                         ):
    # utterance: [[text1,text2,text3... ],[text1,....]]
    assert len(process_dd) == len(process_topic), f'Wrong num: {len(process_dd)} and {len(process_topic)}'
    texts, topics, input_texts = [], [], []
    for utterance in utterances:
        text = 'Utterance:\n'
        for idx, utter in enumerate(utterance[-4:]):
            if idx % 2 == 0:
                text += 'A: ' + utter + '\n'
            else:
                text += 'B: ' + utter + '\n'
        texts.append(text)
    dd_embeddings = calc_embeddings(process_dd)
    batch_texts = split_batch(texts, 512)
    for batch in tqdm(batch_texts):
        simi = cos_sim(calc_embeddings(batch), dd_embeddings)
        indices = torch.topk(simi, 10)[1]
        for main_index in range(len(batch)):
            prompt = ''
            count = prompt_num
            for sub_index in indices[main_index]:
                prompt = prompt + process_topic[int(sub_index)]
                count -= 1
                if count == 0:
                    break
            input_texts.append('Please output the topic of utterance. Please give the answer in English.\n' + prompt + batch[
                main_index] + 'Topic: ')
    batch_input = split_batch(input_texts, batch_size)
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, input_texts)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    for batch in tqdm(batch_input):
        _inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(device)
        if print_tag:
            print(batch[0])
            print(_inputs['input_ids'].shape)
        model.eval()

        # torch.cuda.empty_cache()
        try:
            if 'flan-t5' in model_name:
                results = model.generate(_inputs["input_ids"], attention_mask=_inputs['attention_mask'],
                                         max_length=max_length,
                                         num_beams=num_beams, num_return_sequences=return_num, do_sample=do_sample,
                                         top_k=top_k, top_p=top_p, temperature=temperature)
                results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif 'chatglm' in model_name:
                assert batch_size == 1
                results, history = model.chat(tokenizer, batch[0], history=[])
                results = [str(results.encode('utf-8'))]
            else:
                results = []
        except RuntimeError:
            raise RuntimeError(
                f'CUDA OUT OF MEMORY WHEN GENERATING TOPIC OF UTTERANCE! PLEASE REDUCE PROMPT NUM! CURRENT: {prompt_num}')
        if print_tag:
            print(results[0])
        if file_path is not None:
            with open(file_path, 'a') as f:
                for result in results:
                    f.write(result.replace('Topic: ', '') + '\n')
        topics.extend(results)
    return topics


def batch_event_generate(utterances,
                         model_name,
                         model,
                         tokenizer,
                         device,
                         intent,
                         need,
                         batch_size=16,
                         prompt_num=10,
                         max_length=25,
                         return_num=2,
                         print_tag=False,
                         check=False,
                         do_sample=True,
                         num_beams=1,
                         top_k=0,
                         top_p=0.9,
                         temperature=0.9,
                         file_path_intent=None,
                         file_path_need=None
                         ):
    answer_template = {
        'xNeed': 'speaker needed {}',
        'xIntent': 'speaker wanted {}'
    }

    def batch_sub_function(text, simi, process_texts, process_relations, tag='xIntent', steer='intent: '):
        indices = torch.topk(simi, 500)[1]
        _inputs = []
        for main_index in range(len(text)):
            prompt = ''
            count = prompt_num
            for sub_index in indices[main_index]:
                prompt_answers = trans_str(process_relations[int(sub_index)])
                if not prompt_answers:
                    continue
                prompt += process_texts[int(sub_index)] + '\n' + steer + answer_template[tag].format(
                    ' '.join(prompt_answers[0:return_num])) + '\n'
                count -= 1
                if count == 0:
                    break
            _inputs.append(
                'Please output the intent and need of speaker. Please give the answer in English.\n' + prompt + text[
                    main_index] + '\n' + steer)

        tokenize_inputs = tokenizer(_inputs, return_tensors="pt",
                                    truncation=True,
                                    padding=True,
                                    add_special_tokens=True).to(device)
        if print_tag:
            print(_inputs[0])
            print(_inputs[1])
            print(tokenize_inputs['input_ids'].shape)
        model.eval()

        # torch.cuda.empty_cache()
        try:
            if 'flan-t5' in model_name:
                results = model.generate(tokenize_inputs["input_ids"], attention_mask=tokenize_inputs['attention_mask'],
                                         max_length=max_length,
                                         num_beams=num_beams, num_return_sequences=return_num, do_sample=do_sample,
                                         top_k=top_k, top_p=top_p, temperature=temperature)
                results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif 'chatglm' in model_name:
                assert batch_size == 1
                results, history = model.chat(tokenizer, batch[0], history=[])
                results = [str(results.encode('utf-8'))]
            else:
                results = model.generate(tokenize_inputs["input_ids"], attention_mask=tokenize_inputs['attention_mask'],
                                         max_length=30, num_beams=1,
                                         do_sample=False)
                results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                results[0] = results[0].replace(tokenize_inputs, '')

        except RuntimeError:
            raise RuntimeError(
                f'CUDA OUT OF MEMORY WHEN GENERATING PERSONA OF UTTERANCE! PLEASE REDUCE PROMPT NUM! CURRENT: {prompt_num}')
        if print_tag:
            print(results)
        return results

    index, texts, intent_results, need_results = [], [], [], []
    for utterance in utterances:
        index.append(len(utterance[-4:]))
        for utter in utterance[-4:]:
            texts.append(utter)
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, texts)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    batch_texts = split_batch(texts, batch_size)
    intent_embeddings = calc_embeddings(list(intent['event']))
    need_embeddings = calc_embeddings(list(need['event']))
    for batch in tqdm(batch_texts):
        batch_embeddings = calc_embeddings(batch)
        simi_intent = cos_sim(batch_embeddings, intent_embeddings)
        simi_need = cos_sim(batch_embeddings, need_embeddings)
        batch_intent_results = batch_sub_function(batch, simi_intent, intent['event'], intent['xIntent'], tag='xIntent',
                                                  steer='intent: ')
        batch_need_results = batch_sub_function(batch, simi_need, need['event'], need['xNeed'], tag='xNeed', steer='need: ')
        batch_intent_results = [x.replace('intent: ', '') + ' | ' + y.replace('intent: ', '') for x, y in
                                zip(batch_intent_results[::2], batch_intent_results[1::2])]
        batch_need_results = [x.replace('need: ', '') + ' | ' + y.replace('need: ', '') for x, y in
                              zip(batch_need_results[::2], batch_need_results[1::2])]
        if file_path_intent is not None:
            with open(file_path_intent, 'a') as f:
                f.write(str(batch_intent_results) + '\n')
        if file_path_need is not None:
            with open(file_path_need, 'a') as f:
                f.write(str(batch_need_results) + '\n')
        intent_results.extend(batch_intent_results)
        need_results.extend(batch_need_results)
    return cut_list(intent_results, index), cut_list(need_results, index)


def batch_persona_generate(utterances,
                           model_name,
                           model,
                           tokenizer,
                           device,
                           process_text,
                           process_persona,
                           batch_size=16,
                           prompt_num=5,
                           max_length=50,
                           return_num=2,
                           print_tag=False,
                           check=False,
                           do_sample=True,
                           num_beams=1,
                           top_k=0,
                           top_p=0.9,
                           temperature=0.9,
                           file_path=None,
                           ):
    # use persona chat as prompt dataset
    texts, input_texts, persona_results = [], [], []
    for utterance in utterances:
        texts.append('utterance: ' + ' '.join(utterance[-8:][::2]))  # speaker A
        texts.append('utterance: ' + ' '.join(utterance[-8:][1::2]))  # speaker B
    persona_embeddings = calc_embeddings(process_text)
    batch_texts = split_batch(texts, 256)
    for batch in tqdm(batch_texts):
        simi = cos_sim(calc_embeddings(batch), persona_embeddings)
        indices = torch.topk(simi, 500)[1]
        for main_index in range(len(batch)):
            prompt = ''
            count = prompt_num
            for sub_index in indices[main_index]:
                prompt = prompt + process_persona[int(sub_index)]
                count -= 1
                if count == 0:
                    break
            input_texts.append(
                'Please output the persona of speaker. Please give the answer in English.\n' + prompt + batch[
                    main_index] + 'Persona: ')
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, input_texts)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    batch_input = split_batch(input_texts, batch_size)
    for batch in tqdm(batch_input):
        _inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(device)
        if print_tag:
            print(batch[0])
            print(batch[1])
            print(_inputs['input_ids'].shape)
        model.eval()

        # torch.cuda.empty_cache()
        try:
            if 'flan-t5' in model_name:
                results = model.generate(_inputs["input_ids"], attention_mask=_inputs['attention_mask'],
                                         max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=return_num,
                                         do_sample=do_sample,
                                         top_k=top_k,
                                         top_p=top_p,
                                         temperature=temperature)
                results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif 'chatglm' in model_name:
                assert batch_size == 1
                results, history = model.chat(tokenizer, batch[0], history=[])
                results = [str(results.encode('utf-8'))]
            else:
                results = []
        except RuntimeError:
            raise RuntimeError(
                f'CUDA OUT OF MEMORY WHEN GENERATING PERSONA OF UTTERANCE! PLEASE REDUCE PROMPT NUM! CURRENT: {prompt_num}')
        if print_tag:
            print(results)
            print_tag = False
        results = [x + " | " + y for x, y in zip(results[::2], results[1::2])]
        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write(str(results) + '\n')
        persona_results.extend(results)
    # check persona b
    for i, x in enumerate(texts):
        if x == 'utterance: ':
            persona_results[i] = 'None'

    return persona_results[::2], persona_results[1::2]  # persona A, persona B


def batch_wiki_generate(utterances,
                        model_name,
                        model,
                        tokenizer,
                        device,
                        prompt_dataset,
                        batch_size=16,
                        prompt_num=2,
                        max_length=60,
                        return_num=2,
                        print_tag=False,
                        check=False,
                        do_sample=True,
                        num_beams=1,
                        top_k=0,
                        top_p=0.9,
                        temperature=0.9,
                        file_path=None,
                        ):
    # use Wizard of Wikipedia as prompt dataset
    prompt_text = prompt_dataset['text']
    sample_ids = random.sample(range(len(prompt_text)), 50000)
    sample_texts = [prompt_text[i] for i in sample_ids]
    sample_embeddings = calc_embeddings(sample_texts)

    texts, wikis, input_texts = [], [], []
    # for utterance in utterances:
    #     index.append(len(utterance))
    #     for utter in utterance:
    #         texts.append(utter)
    for utterance in utterances:
        texts.append(' '.join(utterance[-5:]))
    batch_texts = split_batch(texts, 256)
    for batch in tqdm(batch_texts):
        simi = cos_sim(calc_embeddings(batch), sample_embeddings)
        values, indices = torch.topk(simi, 50)
        for main_index in range(len(batch)):
            prompt = ''
            count = prompt_num
            for sub_index in indices[main_index]:
                prompt += 'text: ' + prompt_dataset['text'][int(sub_index)] + '\nWikipedia knowledge: ' + \
                          prompt_dataset['knowledge'][int(sub_index)] + '\n'
                count = count - 1
                if count == 0:
                    break
            input_texts.append(
                'Please output the wiki knowledge of utterance. Please give the answer in English.\n' + prompt + 'text: ' +
                batch[
                    main_index] + '\nWikipedia knowledge: ')
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, input_texts)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    batch_input = split_batch(input_texts, batch_size)

    for batch in tqdm(batch_input):

        tokenize_inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(
            device)
        if print_tag:
            print(batch[0])
            print(tokenize_inputs['input_ids'].shape)
        model.eval()

        # torch.cuda.empty_cache()
        try:
            if 'flan-t5' in model_name:
                results = model.generate(tokenize_inputs["input_ids"], attention_mask=tokenize_inputs['attention_mask'],
                                         max_length=max_length,
                                         num_beams=num_beams, num_return_sequences=return_num, do_sample=do_sample,
                                         top_k=top_k,
                                         top_p=top_p,
                                         temperature=temperature)
                results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            elif 'chatglm' in model_name:
                assert batch_size == 1
                results, history = model.chat(tokenizer, batch[0], history=[])
                results = [str(results.encode('utf-8'))]
            else:
                results = []
        except RuntimeError:
            raise RuntimeError('CUDA OUT OF MEMORY WHEN GENERATING EMOTION OF UTTERANCE! PLEASE REDUCE THE NUM OF PROMPT!')
        sub_1, sub_2 = results[::2], results[1::2]
        results = [i + ' | ' + j for i, j in zip(sub_1, sub_2)]
        if print_tag:
            print(results)
            print_tag = False
        # torch.cuda.empty_cache()
        if file_path is not None:
            with open(file_path, 'a') as f:
                for result in results:
                    f.write(result + '\n')
        wikis.extend(results)
    return wikis


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
    strings = string.split(']')
    lists = []
    for s in strings:
        if len(s) <= 1 or s[0] != '[':
            continue
        s = eval(s + ']')
        lists.extend(s)
    return lists


def inside_template(utterance, persona_A, persona_B, topic, wiki=None):
    background_knowledge = "Please generate a response based on knowledge below. Please keep the response diverse. Please give the answer in English.\nBackground Knowledge: Topic: " + \
                           topic
    background_knowledge = background_knowledge + "[SEP]Persona of A: " + persona_A if persona_A is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Persona of B: ' + persona_B if persona_B is not None else background_knowledge
    if wiki is not None:
        background_knowledge += '[SEP]Wikipedia Knowledge: ' + wiki + '\nDialogue: \n'
    else:
        background_knowledge = background_knowledge + '\nDialogue: \n'
    text = tag = ''
    for utter in utterance:
        tag = utter['speaker']
        if 'xIntent' not in utter.keys():
            text += utter['speaker'] + ": " + utter['text'] + '\n'
        else:
            text += "emotion: " + utter['emotion'] + ' | intent: ' + utter['xIntent'] + ' | need: ' + utter['xNeed'] + \
                    ' | ' + utter['speaker'] + ": " + utter['text'] + '\n'
    text = background_knowledge + text
    text = text + 'B: ' if tag == 'A' else text + "A: "
    return text


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


def ablation_template(utterance, persona_A=None, persona_B=None, topic=None, wiki=None):
    emotion_A = utterance[-1]['emotion'] if 'emotion' in utterance[-1].keys() else None
    emotion_B = utterance[-2]['emotion'] if 'emotion' in utterance[-2].keys() else None
    intent_A = utterance[-1]['intent'] if 'intent' in utterance[-1].keys() else None
    need_A = utterance[-1]['need'] if 'need' in utterance[-1].keys() else None
    intent_B = utterance[-2]['intent'] if 'intent' in utterance[-2].keys() else None
    need_B = utterance[-2]['need'] if 'need' in utterance[-2].keys() else None

    # background_knowledge = "Please generate a response based on knowledge below, including persona knowledge, emotion knowledge, topic knowledge, \
    # intent and need knowledge ( describes what do speakers want to do ). Please keep the response diverse.\
    # \nBackground Knowledge: Persona of A: " + persona_A
    background_knowledge = "Please generate a response based on background knowledge below. Please keep the response diverse.\nBackground Knowledge: "
    background_knowledge = background_knowledge + "Topic: " + topic if topic is not None else background_knowledge
    background_knowledge = background_knowledge + "[SEP]Persona of A: " + persona_A if persona_A is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Persona of B: ' + persona_B if persona_B is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Emotion of A: ' + emotion_A if emotion_A is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Emotion of B: ' + emotion_B if emotion_B is not None else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Intent of A: ' + intent_A + '[SEP]Need of A: ' + need_A if intent_A is not None \
        else background_knowledge
    background_knowledge = background_knowledge + '[SEP]Intent of B: ' + intent_B + '[SEP]Need of B: ' + need_B if intent_B is not None \
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


def batch_reply_generate(utterances,
                         model_name,
                         model,
                         tokenizer,
                         device,
                         batch_size=16,
                         max_length=600,
                         min_length=10,
                         return_num=1,
                         print_tag=False,
                         check=False,
                         do_sample=True,
                         num_beams=1,
                         top_k=0,
                         top_p=0.95,
                         temperature=0.9,
                         file_path=None,
                         ):
    # utterances: [text1,text2,text3,....]
    if check:
        max_len, over_rate, avg_len, total_num = check_token_len(tokenizer, utterances)
        print("max token length: ", max_len)
        print('over 512 rate: ', over_rate)
        print('average length: ', avg_len)
        print('total num: ', total_num)
        continue_command = input('Would you want to continue generation? [Y/N]')
        if continue_command == 'Y' or continue_command == 'y':
            pass
        else:
            print("end generation!")
            sys.exit()
    batch_texts = split_batch(utterances, batch_size)
    end_token_id = tokenizer('\n', return_tensors="pt")['input_ids'].shape[-1]
    if print_tag:
        print("end token id: ", end_token_id)
    replies = []
    for batch in tqdm(batch_texts):
        _input = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(device)

        model.eval()
        if print_tag:
            print(batch[0])
            print(_input['input_ids'].shape)

        # torch.cuda.empty_cache()
        if 'flan-t5' in model_name:
            results = model.generate(_input["input_ids"],
                                     attention_mask=_input['attention_mask'],
                                     max_length=max_length,
                                     min_length=min_length,
                                     num_beams=num_beams,
                                     num_return_sequences=return_num,
                                     do_sample=do_sample,
                                     #  eos_token_id=end_token_id,  # '\n'
                                     top_k=top_k,
                                     top_p=top_p,
                                     early_stopping=True,
                                     repetition_penalty=1.03,
                                     no_repeat_ngram_size=3,
                                     temperature=temperature)
            results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        elif 'gpt-neox' in model_name:
            results = model.generate(_input["input_ids"],
                                     attention_mask=_input['attention_mask'],
                                     max_length=max_length + _input['input_ids'].shape[-1],
                                     min_length=min_length + _input['input_ids'].shape[-1],
                                     num_beams=num_beams,
                                     num_return_sequences=return_num,
                                     do_sample=do_sample,
                                     eos_token_id=187,
                                     top_k=top_k,
                                     top_p=top_p,
                                     early_stopping=True,
                                     # repetition_penalty=1.03,
                                     temperature=temperature)
            results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for idx in range(len(results)):
                results[idx] = results[idx].replace(batch[idx], '')
                results[idx] = results[idx].split('\n')[0]
        elif 'opt' in model_name:
            results = model.generate(_input["input_ids"],
                                     attention_mask=_input['attention_mask'],
                                     max_length=max_length + _input['input_ids'].shape[-1],
                                     min_length=min_length + _input['input_ids'].shape[-1],
                                     num_beams=num_beams,
                                     num_return_sequences=return_num,
                                     do_sample=do_sample,
                                     eos_token_id=50118,
                                     top_k=top_k,
                                     top_p=top_p,
                                     early_stopping=True,
                                     # repetition_penalty=1.03,
                                     temperature=temperature)
            results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for idx in range(len(results)):
                results[idx] = results[idx].replace(batch[idx], '')
                results[idx] = results[idx].replace('\n', ' <s> ')
                results[idx] = "<BOS>" + results[idx] + "<EOS>"
        elif 'chat' in model_name:
            assert batch_size == 1
            results, history = model.chat(tokenizer, batch[0], history=[])
            results = [str(results.encode('utf-8')).replace("b'", '')[:-1]]
        else:
            results = []
        if print_tag:
            print(results)
            print_tag = False
        if file_path is not None:
            with open(file_path, 'a') as f:
                for result in results:
                    f.write(result.replace('A: ', '').replace('B: ', '') + '\n')
        replies.extend(results)
    return replies
