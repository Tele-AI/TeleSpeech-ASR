#!/bin/bash

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch

export CUDA_VISIBLE_DEVICES="0,1,2,3"
stage=5 # train:5, decode:6
stop_stage=6

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

token_type=char
# bpemode (unigram or bpe)
nbpe=6000
bpemode=unigram

data_type=ark
data_set=wenetspeech_test
train_set=train_s
valid_set=dev
recog_set="dev test_meeting test_net"
pretrain_model=d2v2

data_path=/home/${data_set}
dir=exp/${pretrain_model}_${data_type}_conformer
train_config=conf/train_${pretrain_model}_${data_type}_conformer.yaml

if [ "${token_type}" = bpe ]; then
    dict=${data_path}/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
    bpemodel=${data_path}/lang_char/${train_set}_${bpemode}${nbpe}.model
elif [ "${token_type}" = char ]; then
    dict=${data_path}/${train_set}/lang_char.txt
    bpemodel=None
else
    echo "Error: not supported token_type"
    exit 0
fi

checkpoint=
cmvn=false
num_workers=4
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="ctc_greedy_search"

# Specify decoding_chunk_size if it's a unified dynamic chunk trained model
# -1 for full chunk
decoding_chunk_size=
ctc_weight=0.3
reverse_weight=0.0

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # Training
    mkdir -p $dir
    INIT_FILE=$dir/ddp_init
    rm -f $INIT_FILE # delete old one before starting
    #init_method=file://$(readlink -f $INIT_FILE)
    init_method="tcp://localhost:31538"
    echo "$0: init method is $init_method"
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Use "nccl" if it works, otherwise use "gloo"
    dist_backend="nccl"
    world_size=`expr $num_gpus \* $num_nodes`
    echo "total gpus is: $world_size"
    cmvn_opts=
    $cmvn && cp ${data_path}/${train_set}/global_cmvn $dir
    $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
    # train.py will write $train_config to $dir/train.yaml with model input
    # and output dimension, train.yaml will be used for inference or model
    # export later
    for ((i = 0; i < $num_gpus; ++i)); do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
        rank=`expr $node_rank \* $num_gpus + $i`
        python wenet/bin/train.py --gpu $gpu_id \
            --config $train_config \
            --data_type $data_type \
            --symbol_table $dict \
            --bpe_model $bpemodel \
            --train_data $data_path/$train_set/data.list \
            --cv_data $data_path/$valid_set/data.list \
            ${checkpoint:+--checkpoint $checkpoint} \
            --model_dir $dir \
            --ddp.init_method $init_method \
            --ddp.world_size $world_size \
            --ddp.rank $rank \
            --ddp.dist_backend $dist_backend \
            --num_workers ${num_workers} \
            --prefetch ${prefetch} \
            $cmvn_opts || exit 1;
    } &
    done
    wait
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # Test model, please specify the model you want to test by --checkpoint
    # cmvn_opts=
    # $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
    if [ ${average_checkpoint} == true ]; then
        decode_checkpoint=$dir/avg_${average_num}.pt
        echo "do model average and final checkpoint is $decode_checkpoint"
        python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --val_best || exit 1;
    fi

    gpu_index=0
    for test in ${recog_set}; do
    {
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$gpu_index+1])
        for mode in ${decode_modes}; do
        {
            #gpu_id=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print $1}')
            test_dir=$dir/${test}_${mode}
            mkdir -p $test_dir
            python wenet/bin/recognize.py --gpu $gpu_id \
            --mode $mode \
            --config $dir/train.yaml \
            --data_type $data_type \
            --bpe_model $bpemodel \
            --test_data $data_path/$test/data.list \
            --checkpoint $decode_checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $ctc_weight \
            --reverse_weight $reverse_weight \
            --result_file $test_dir/text_ori \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} || exit 1;
            if [ "${token_type}" = bpe ]; then
                tools/spm_decode --model=${bpemodel} --input_format=piece < $test_dir/text_ori | sed -e "s/▁/ /g" > $test_dir/text
                python tools/compute-wer.py --char=0 --v=1 \
                $data_path/$test/text $test_dir/text_ori > $test_dir/wer
            elif [ "${token_type}" = char ]; then
                python tools/compute-wer.py --char=1 --v=1 \
                $data_path/$test/text $test_dir/text_ori > $test_dir/wer
            fi
        } &
        done
        gpu_index=$((gpu_index + 1))
    }
    done
    wait
fi
