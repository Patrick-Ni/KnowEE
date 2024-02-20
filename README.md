# KnowEE
Code for EMNLP2023 “Multi-Source Multi-Type Knowledge Exploration and Exploitation for Dialogue Generation”

### 1. Environment

- CUDA >= 11.6
- torch >= 2.0.0

```shell
pip install -r requirements.txt
```



### 2. Data

data/preprocessed_data



### 3.Models

download [Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl), [ChatGLM](https://huggingface.co/THUDM/chatglm-6b), [GPT-NEOX](https://github.com/EleutherAI/gpt-neox) from huggingface and change the model path in ``src/load_models_and_datasets.py``



### 4. Run

```
sh src/run.sh
```



If you are interested in our work, please cite:

```
@inproceedings{conf/emnlp/NiDRL23,
  author       = {Xuanfan Ni and
                  Hongliang Dai and
                  Zhaochun Ren and
                  Piji Li},
  title        = {Multi-Source Multi-Type Knowledge Exploration and Exploitation for
                  Dialogue Generation},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023, Singapore, December 6-10, 2023},
  year         = {2023},
  url          = {https://aclanthology.org/2023.emnlp-main.771},
}
```

