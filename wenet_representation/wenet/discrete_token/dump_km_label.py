# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import argparse
import joblib
import torch
import tqdm
import numpy as np
from wenet.discrete_token.utils import read_npy_feat, read_kaldi_feat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def dump_label(feat_dir, subset, km_model_path, rank, nshard, lab_dir, input_type):
    apply_kmeans = ApplyKmeans(km_model_path)

    if input_type == "npy":
        generator, num, utts = read_npy_feat(os.path.join(feat_dir, subset), rank, -1, use_iterate=True)
    elif input_type == "kaldi_ark":
        generator, num, utts = read_kaldi_feat(os.path.join(feat_dir, subset), rank, -1, use_iterate=True)
    iterator = generator()

    lab_path = f"{os.path.join(lab_dir, subset)}/feat.{rank}.km"
    os.makedirs(os.path.join(lab_dir, subset), exist_ok=True)
    logger.info(f"doing {os.path.join(lab_dir, subset)}/feat.{rank}.km")
    with open(lab_path, "w") as f:
        for feat, utt in zip(tqdm.tqdm(iterator, total=num), utts):
            lab = (apply_kmeans(feat) + 1).tolist()  # 0 for padding in training
            f.write(utt + "\t" + " ".join(map(str, lab)) + "\n")

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--km_model_path", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--lab_dir")
    parser.add_argument(
        "--input_type",
        type=str,
        default="npy",
        choices=["npy", "kaldi_ark"],
    )

    args = parser.parse_args()
    logging.info(str(args))
    dump_label(**vars(args))