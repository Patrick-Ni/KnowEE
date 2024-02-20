#for ((i=1; i<=100; i ++))
#do
##  row='../results/base_result/gen_normal:'$i'.txt'
#  row='../results/saved_results/event_ver/topk/output:'$i'.txt'
##  ans=empathetic_answers.txt
#  ans='../results/saved_results/event_ver/topk/refer:'$i'.txt'
#  CUDA_VISIBLE_DEVICES=1 python eval.py \
#  --generation_file  $row \
#  --source_file $ans \
#  --eval_output eval_results/event_topk_results.json \
#  --ppl --rouge --bleu --dist --bert_score --meteor
#  python ../kill_python.py --progm_name eval
#done
#row='../results/saved_results/event_ver/beam/.txt'
#  ans=empathetic_answers.txt
#ans='../results/saved_results/event_ver/refer:beam.txt'
#python eval.py \
#--generation_file  $row \
#--source_file $ans \
#--ppl --rouge --bleu --dist --bert_score --meteor
#python ../kill_python.py --progm_n ame eval

#for ((i=1; i<=100; i ++))
#do
##  row='../results/base_result/gen_normal:'$i'.txt'
#  row='../results/saved_results/emo_ver/beam/output:'$i'.txt'
##  ans=empathetic_answers.txt
#  ans='../results/saved_results/emo_ver/beam/refer:'$i'.txt'
#  CUDA_VISIBLE_DEVICES=1 python eval.py \
#  --generation_file  $row \
#  --source_file $ans \
#  --eval_output eval_results/emo_beam_results.json \
#  --ppl --rouge --bleu --dist --bert_score --meteor
#  python ../kill_python.py --progm_name eval
#done
#CUDA_VISIBLE_DEVICES=2 python eval.py \
#  --generation_file ../results/save_results/base/model_30000_.tar_output.txt \
#  --source_file ../results/save_results/base/model_30000_.tar_output.txt \
#  --eval_output eval_results/base_greedy_results.json \
#  --ppl --rouge --bleu --dist --bert_score --meteor

#for ((i=101; i<=150 ; i ++))
#do
#  gen='../results/save_results/odkg_base/greedy/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_base/greedy/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_base_greedy_results.json
#  python ../kill_python.py --progm_name eval.py
#  gen='../results/save_results/odkg_base/beam/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_base/beam/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_base_beam_results.json
#  python ../kill_python.py --progm_name eval.py
#  gen='../results/save_results/odkg_base/topk/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_base/topk/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_base_topk_results.json
#  python ../kill_python.py --progm_name eval.py
#done
#
#for ((i=101; i<=139 ; i ++))
#do
#  gen='../results/save_results/odkg_raw/greedy/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_raw/greedy/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_raw_greedy_results.json
#  python ../kill_python.py --progm_name eval.py
#  gen='../results/save_results/odkg_raw/beam/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_raw/beam/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_raw_beam_results.json
#  python ../kill_python.py --progm_name eval.py
#  gen='../results/save_results/odkg_raw/topk/model_'$((i*500))'_output.txt'
#  ref='../results/save_results/odkg_raw/topk/model_'$((i*500))'_refer.txt'
#  CUDA_VISIBLE_DEVICES=6 /data/xfni/anaconda3/bin/python eval.py \
#  --generation_file $gen \
#  --source_file $ref \
#  --eval_output eval_results/odkg_raw_topk_results.json
#  python ../kill_python.py --progm_name eval.py
#done
# gen='../results/opt-30b/empathetic/baseline.txt'


# ref='../baseline/PFL/bart_odkg_ans.txt'
# gen='../baseline/PFL/bart_on_odkg_output.txt'
# ref='../baseline/FSB/odkg_neo_ans.txt'
# gen="../baseline/FSB/odkg_gpt-neo.txt"
#CUDA_VISIBLE_DEVICES=4 /data/xfni/anaconda3/bin/python eval.py \
#--generation_file $gen \
#--source_file $ref \
#--eval_output eval_results/ed_final.json

gen='/data/xfni/code/KnowEE/baseline/MSDP/output_response.txt'
# gen='../baseline/pretrained_dialogue_models/DialoGPT-large_copy/dialydialog_output.txt'
# ref='../baseline/FSB/pc_ans.txt'
ref='/data/xfni/code/KnowEE/baseline/MSDP/answers.txt'
CUDA_VISIBLE_DEVICES=3 /data/xfni/anaconda3/bin/python eval.py \
--generation_file $gen \
--source_file $ref \
--eval_output ee.txt \
--lang en --ppl --dist --ignore