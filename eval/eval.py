# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2022/3/19 22:11
# @Author : Patrick Ni
# @File : metrics.py
# @description:
import warnings

# os.environ['CUDA_VISIBLE_DEVICES']='4'
import jieba

warnings.filterwarnings("ignore")
from rouge_chinese import Rouge as cRG
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import numpy as np

# from itertools import chain
# import nltk
# from nlg eval import compute_metrics

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
import nltk
from datasets import load_metric

metric_path = '/data/xfni/code/metrics/'


def _strip(s):
    return s.strip()


def compute_metrics_from_results(hypothesis, references, no_overlap=False, no_skipthoughts=False, no_glove=False):
    hyp_list = hypothesis
    ref_list = [list(map(_strip, ref_s)) for ref_s in zip(*[references])]
    ref_s = {idx: stripped_lines for (idx, stripped_lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(ref_s) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref_s, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.6f" % (m, sc))
                    ret_scores[m] = sc
            else:
                # print("%s: %0.6f" % (method, score))
                ret_scores[method] = score
            if isinstance(scorer, Meteor):
                scorer.close()
        del scorers

    if not no_skipthoughts:
        from nlgeval.skipthoughts import skipthoughts
        from sklearn.metrics.pairwise import cosine_similarity

        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        vector_hyps = encoder.encode([h.strip() for h in hyp_list], verbose=False)
        ref_list_T = np.array(ref_list).T.tolist()
        vector_refs = map(lambda ref_l: encoder.encode([r.strip() for r in ref_l], verbose=False), ref_list_T)
        cosine_similarity = list(map(lambda ref_v: cosine_similarity(ref_v, vector_hyps).diagonal(), vector_refs))
        cosine_similarity = np.max(cosine_similarity, axis=0).mean()
        print("SkipThoughtsCosineSimilarity: %0.6f" % cosine_similarity)
        ret_scores['SkipThoughtCS'] = cosine_similarity
        del model

    if not no_glove:
        from nlgeval.word2vec.evaluate import eval_emb_metrics

        glove_hyps = [h.strip() for h in hyp_list]
        ref_list_T = np.array(ref_list).T.tolist()
        glove_refs = map(lambda ref_l: [r.strip() for r in ref_l], ref_list_T)
        scores = eval_emb_metrics(glove_hyps, glove_refs)
        # print(scores)
        scores = scores.split('\n')
        for score in scores:
            name, value = score.split(':')
            value = float(value.strip())
            ret_scores[name] = value

    return ret_scores


def get_dist(file, lang='en'):
    res = {}
    itr = 0
    for sentence in file:
        res[itr] = nltk.word_tokenize(sentence) if lang == 'en' else list(jieba.cut(sentence))
        itr += 1
    uni_grams = []
    bi_grams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        uni_grams += ugs
        bi_grams += bgs
        ma_dist1 += len(set(ugs)) / float(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / float(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(uni_grams)) / float(len(uni_grams) + 1e-16)
    mi_dist2 = len(set(bi_grams)) / float(len(bi_grams) + 1e-16)
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


def load_files_with_special_symbols(file_path):
    files = []
    with open(file_path, 'r', encoding='utf-8') as wf:
        text = ''
        for _l in wf.readlines():
            text = text + _l
            if '<EOS>' in text:
                files.append(text.replace('<EOS>\n', '').replace('<BOS>', ''))
                text = ''
    return files


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 100)
    print('Opts'.center(100))
    print('-' * 100)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(100))
    print('=' * 100)


def compute_chinese_bleu_and_rouge(predictions, references):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-1": [],
        "bleu-2": [],
        "bleu-3": [],
        "bleu-4": []
    }
    for pred, label in zip(predictions, references):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        chinese_rouge = cRG()
        scores = chinese_rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]
        for k, v in result.items():
            score_dict[k].append(v["f"])
        score_dict["bleu-1"].append(sentence_bleu([list(label)], list(pred), weights=[1, 0, 0, 0], smoothing_function=SmoothingFunction().method3))
        score_dict["bleu-2"].append(
            sentence_bleu([list(label)], list(pred), weights=[0.5, 0.5, 0, 0], smoothing_function=SmoothingFunction().method3))
        score_dict["bleu-3"].append(
            sentence_bleu([list(label)], list(pred), weights=[0.33, 0.33, 0.33, 0], smoothing_function=SmoothingFunction().method3))
        score_dict["bleu-4"].append(sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    score_dict['bleu'] = 0.25 * (score_dict["bleu-1"] + score_dict["bleu-2"] + score_dict["bleu-3"] + score_dict["bleu-4"])
    return score_dict


def main():
    ppl = load_metric(metric_path + 'perplexity', model_id='gpt2') if args.ppl else None
    bert_score = load_metric(metric_path + 'bertscore', model_path='/data/xfni/code/Pretrained_Models/bert-base-uncased',
                             model_type="bert-base-uncased", num_layers=8) if args.bertscore else None
    rouge = load_metric(metric_path + 'rouge') if args.rouge else None

    preds, refs = [], []
    if not args.symbol:
        with open(args.generation_file, 'r') as f:
            for line in f.readlines():
                preds.append(line.replace('\n', ''))

    else:
        preds = load_files_with_special_symbols(args.generation_file)
    with open(args.source_file, 'r') as f:
        for line in f.readlines():
            refs.append(line.replace('\n', ''))
    if args.ignore:
        refs = refs[0:len(preds)]
    print(len(refs))
    metric_score_dict = dict({'File Name': args.generation_file})
    if args.ppl:
        metric_score_dict['PPL'] = ppl.compute(input_texts=preds, model_id='/data/xfni/code/Pretrained_Models/gpt2')['mean_perplexity']
    if args.lang == 'en':
        if args.bleu:
            bleu_score = compute_metrics_from_results(hypothesis=preds, references=refs, no_skipthoughts=True, no_glove=True)
            bleu_1, bleu_2, bleu_3, bleu_4 = bleu_score['Bleu_1'], bleu_score['Bleu_2'], bleu_score['Bleu_3'], bleu_score['Bleu_4']
            metric_score_dict.update(
                {'BLEU': 0.25 * (bleu_3 + bleu_2 + bleu_4 + bleu_1), 'BLEU-1': bleu_1, 'BLEU-2': bleu_2, 'BLEU-3': bleu_3, 'BLEU-4': bleu_4})

        if args.rouge:
            rouge_score = rouge.compute(predictions=preds, references=refs)
            metric_score_dict.update(
                {'Rouge-1': rouge_score['rouge1'][1][2], 'Rouge-2': rouge_score['rouge2'][1][2], 'Rouge-L': rouge_score['rougeL'][1][2],
                 'Rouge-S': rouge_score['rougeLsum'][1][2]})
    else:
        score_dict = compute_chinese_bleu_and_rouge(preds, refs)
        if args.bleu:
            metric_score_dict.update(
                {'BLEU': score_dict['bleu'], 'BLEU-1': score_dict['bleu-1'], 'BLEU-2': score_dict['bleu-2'], 'BLEU-3': score_dict['bleu-3'],
                 'BLEU-4': score_dict['bleu-4']})
        if args.rouge:
            metric_score_dict.update({'Rouge-1': score_dict['rouge-1'], 'Rouge-2': score_dict['rouge-2'], 'Rouge-L': score_dict['rouge-l']})
    if args.bertscore:
        bert_scores = bert_score.compute(predictions=preds, references=refs, lang='en')['f1']
        metric_score_dict['BERTScore'] = sum(bert_scores) / len(bert_scores)
        # print(rouge_score)
    if args.dist:
        _, _, d1, d2, _ = get_dist(preds, lang=args.lang)
        metric_score_dict.update({'Dist-1': d1, 'Dist-2': d2})

    # print(metrics_dict)
    with open(args.eval_output, 'a') as f:
        for k, v in metric_score_dict.items():
            if k == 'File Name':
                print("{}: {}".format(k, v))
                f.write("{}: {}".format(k, v) + '\n')
            elif k == 'PPL':
                print("{}: {:.2f}".format(k, v))
                f.write("{}: {:.2f}".format(k, v) + '\n')
            else:
                print("{}: {:.2f}".format(k, float(v) * 100))
                f.write("{}: {:.2f}".format(k, float(v) * 100) + '\n')
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--generation_file', type=str, required=True, default=None)
    parser.add_argument('--source_file', type=str, required=True, default=None)
    # eval settings
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument('--bleu', action='store_true')
    parser.add_argument('--rouge', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--bertscore', action='store_true')

    # file settings
    parser.add_argument('--eval_output', type=str, default='2')
    parser.add_argument('--symbol', action='store_true')
    parser.add_argument('--ignore', action='store_true')
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'cn'])
    args = parser.parse_args()

    print_opts(args)
    main()
