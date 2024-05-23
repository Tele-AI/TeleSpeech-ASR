. ./path.sh || exit 1

pretrained_model=/path/to/model
python  /path/to/fairseq/fairseq_cli/hydra_train.py -m --config-dir config/v2_dialect_asr \
    --config-name base_audio_finetune_140h \
    common.user_dir=/path/to/data2vec_dialect \
    model.w2v_path=${pretrained_model} \
    task.data=/path/to/data
