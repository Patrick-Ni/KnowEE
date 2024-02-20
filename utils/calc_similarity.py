# /data/xfni/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/2/27 14:03
# @Author       : patrick
# @File         : calc_similarity.py
# @Description  :
# /data/xfni/code/anaconda/bin/python
# -*- coding: utf-8 -*-
# @Time         : 2023/2/15 20:21
# @Author       : patrick
# @File         : calc_similarity.py
# @Description  : a program for testing sentence transformer
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert
import torch

# model = SBert('paraphrase-multilingual-MiniLM-L12-v2') #如果这调用模型有问题，需自行下载，该模型 ，
# [下载网址](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/)

model = SBert("all-MiniLM-L12-v2")


def test():
    # Two lists of sentences
    sentences1 = ['The cat sits outside',
                  'A man is playing guitar',
                  'The new movie is awesome']
    # sentences1 = ['I love you!!', 'fuck you!', 'I love you!!']
    sentences2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']
    # sentences2 = []
    # for i in range(30):
    #     sentences2.append('The dog plays in the garden')
    # sentences2 = 'I hate you!!'
    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Compute cosine-similarities
    cosine_scores = cos_sim(embeddings1, embeddings2)
    print(torch.topk(cosine_scores, k=3)[1])
    print(torch.topk(cosine_scores, k=3)[1][0][2])


def calc_sentence_similarity(sentence_list_1, sentence_list_2):
    # multi-sentences: n-list1, m-list2, return tensor with shape(nxm)
    # single-sentences: return tensor with shape 1x1
    embeddings1 = model.encode(sentence_list_1)
    embeddings2 = model.encode(sentence_list_2)
    return cos_sim(embeddings1, embeddings2)


def calc_embeddings(sentence_list):
    return model.encode(sentence_list)


if __name__ == '__main__':
    test()
