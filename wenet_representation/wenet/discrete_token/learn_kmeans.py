# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import yaml
import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib
from wenet.discrete_token.utils import read_npy_feat, read_kaldi_feat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_path, rank, percent, input_type):
    if input_type == "npy":
        return read_npy_feat(feat_path, rank, percent)
    elif input_type == "kaldi_ark":
        return read_kaldi_feat(feat_path, rank, percent)

def load_feature(feat_path, nshard, percent, input_type):
    assert percent <= 1.0
    if nshard==1:
        feat = load_feature_shard(feat_path, 0, percent, input_type)
    else:
        feat = np.concatenate(
            [
                load_feature_shard(feat_path, r, percent, input_type)
                for r in range(nshard)
            ],
            axis=0,
        )
    logging.info(f"Successfully loaded feature with dimension {feat.shape}")
    return feat

def learn_kmeans(
    feat_dir,
    train_set,
    km_model_path,
    nshard,
    config,
    percent,
    verified,
    input_type,
    seed
):
    np.random.seed(seed)
    with open(config, "r") as fin:
        conf = yaml.load(fin, Loader=yaml.FullLoader)
    kmeans_conf = conf.get("kmeans_conf", None)
    feat_path = os.path.join(feat_dir, train_set)
    feat = load_feature(feat_path, nshard, percent, input_type)

    logger.info("dump kmeans as usual")
    km_model = get_km_model(**kmeans_conf)
    km_model.fit(feat)
    joblib.dump(km_model, km_model_path)
    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str)
    parser.add_argument("--train_set", type=str, default="train")
    parser.add_argument("--km_model_path", type=str)
    parser.add_argument("--nshard", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--percent", type=float, default=-1)
    parser.add_argument("--verified", type=str, default="false")
    parser.add_argument(
        "--input_type",
        type=str,
        default="npy",
        choices=["kaldi_ark", "npy"],
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    args.verified = args.verified.lower() == "true"
    logging.info(str(args))
    learn_kmeans(**vars(args))
