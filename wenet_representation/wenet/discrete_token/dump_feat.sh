#!/bin/bash

. ./path.sh || exit 1;

stage=0
stop_stage=0
root_dir=/path/to/data  # data root directory
dataset="train_s dev" # dataset
train_km_set="train_s"  # kmeans training set
nshard=1
feat_dir=${root_dir}/dump_d2v_discrete  # will make this dir if not exists
config=kmeans_d2v.yaml

km_model_path=${feat_dir}/km_model
lab_dir=${feat_dir}
percent=-1
verified=false

input_type=wenet_ark  # [raw, wenet_ark]
feat_save_type=npy  # [npy, kaldi_ark]

remove_duplicate=true

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    for ((rank = 0; rank < ${nshard}; ++rank)); do
    {
        for subset in ${dataset}; do
            python wenet/discrete_token/dump_pretrain_feature.py \
                --root_dir ${root_dir} \
                --subset ${subset} \
                --rank ${rank} \
                --nshard ${nshard} \
                --config ${config} \
                --feat_dir ${feat_dir}/${subset} \
                --input_type ${input_type} \
                --output_type ${feat_save_type} || exit 1;
        done
    } &
    done
    wait
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    python wenet/discrete_token/learn_kmeans.py \
        --feat_dir ${feat_dir} \
        --train_set ${train_km_set} \
        --km_model_path ${km_model_path} \
        --nshard ${nshard} \
        --config ${config} \
        --percent ${percent} \
        --verified ${verified} \
        --input_type ${feat_save_type} || exit 1;

    echo "learn_kmeans done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for ((rank = 0; rank < ${nshard}; ++rank)); do
    {
        for subset in ${dataset}; do
            python wenet/discrete_token/dump_km_label.py \
                --feat_dir ${feat_dir} \
                --subset ${subset} \
                --km_model_path ${km_model_path} \
                --rank ${rank} \
                --nshard ${nshard} \
                --lab_dir ${lab_dir} \
                --input_type ${feat_save_type} || exit 1;
        done
    } &
    done
    wait
    echo "dump km lables done, start processing labels"

    for subset in ${dataset}; do
        # merge nshard km if exist
        cat ${lab_dir}/${subset}/feat.*.km > ${lab_dir}/${subset}/feat.km
        rm ${lab_dir}/${subset}/feat.*.km
        # remove duplicate
        if [ "$remove_duplicate" = true ]; then
            awk '{
                printf "%s\t%s", $1, $2;
                for (i = 3; i <= NF; i++) {
                    if ($i != $(i-1)) {
                        printf " %s", $i;
                    }
                }
                printf "\n";
            }' ${lab_dir}/${subset}/feat.km > ${lab_dir}/${subset}/feat.ddup.km
            mv ${lab_dir}/${subset}/feat.ddup.km ${lab_dir}/${subset}/feat.km
        fi
        # change data.list by feat.km
        if [ -L "${root_dir}/${subset}/data.list.discrete" ]; then
            rm ${root_dir}/${subset}/data.list.discrete
        fi
        python wenet/discrete_token/change_datalist.py \
            --subset ${subset} \
            --lab_dir ${lab_dir} \
            --datalist_dir ${root_dir} \
            --ori_type ${input_type} || exit 1;
        ln -s ${lab_dir}/${subset}/data.list.discrete ${root_dir}/${subset}/data.list.discrete
    done
fi