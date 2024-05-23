. ./path.sh || exit 1

data=/path/to/data
model=/path/to/model
result_path=/path/to/decode_result
python infer.py \
    --config-dir config \
    --config-name infer \
    task=spec_finetuning \
    task.data=${data} \
    task.normalize=false \
    common.user_dir=/path/to/data2vec_dialect \
    common_eval.path=${model} \
    common_eval.results_path=${result_path} \
    common_eval.quiet=false \
    dataset.gen_subset=train
  