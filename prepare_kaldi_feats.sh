#!/bin/bash

. path.sh || exit 1;

stage=0
stop_stage=1

cmd=run.pl
mfcc_config=mfcc_hires.conf
datadir=/path/to/data  # need wav.scp
nj=8
split_dir=$datadir/split${nj}utt

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # split wav.scp into $nj
    mkdir -p $split_dir
    split_scps=
    scp=$datadir/wav.scp
    for n in $(seq $nj); do
        mkdir -p $split_dir/$n
        split_scps="$split_scps $split_dir/$n/wav.scp"
    done
    utils/split_scp.pl $scp $split_scps || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for data in $datadir; do
        src=$split_dir
        dir=$data/feat
        mkdir -p $dir
        mkdir -p $dir/log

        utils/split_data.sh --per-utt $1 $nj

        if [ -f ${src}/1/segments ]; then
        $cmd JOB=1:${nj} $dir/log/feats.JOB.log \
                extract-segments scp,p:${src}/JOB/wav.scp ${src}/JOB/segments ark:- \| \
                compute-mfcc-feats --verbose=2 \
                --config=$mfcc_config ark:- ark:- \| \
                copy-feats --compress=true ark:- \
                ark,scp:$dir/mfcc.JOB.ark,$dir/mfcc.JOB.scp
        else
        $cmd JOB=1:${nj} $dir/log/feats.JOB.log \
                compute-mfcc-feats --verbose=2 \
                --config=$mfcc_config scp,p:${src}/JOB/wav.scp ark:- \| \
                copy-feats --compress=true ark:- \
                ark,scp:$dir/mfcc.JOB.ark,$dir/mfcc.JOB.scp
        fi
        for n in $(seq $nj); do
            feat-to-len scp,p:$dir/mfcc.$n.scp ark,t:$dir/feat2len.$n.txt || exit 1;
        done
        cat $dir/mfcc.*.scp > $dir/mfcc.scp
        cat $dir/feat2len.*.txt > $dir/feat2len.txt
    done
fi