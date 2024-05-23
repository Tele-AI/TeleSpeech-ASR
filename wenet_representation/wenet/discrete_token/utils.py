import os
import sys
import logging
import numpy as np
import kaldiio
from npy_append_array import NpyAppendArray

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


class NpyWriter(object):
    def __init__(self, feat_path, rank):
        os.makedirs(feat_path, exist_ok=True)

        npy_file = f"{feat_path}/feats.{rank}.npy"
        lens_file = f"{feat_path}/feats.{rank}.len"
        if os.path.exists(npy_file):
            logger.info(f"npy file {npy_file} exist, remove the old one")
            os.remove(npy_file)
        self.feat_path = NpyAppendArray(npy_file)
        self.leng_path = open(lens_file, "w", encoding="utf-8")

    def close(self):
        try:
            self.leng_path.close()
        except Exception as e:
            print("Error! close leng_path failed")
            raise e

    def save_feats(self, feat, utt):
        self.feat_path.append(feat)
        self.leng_path.write(f"{utt}\t{len(feat)}\n")

class KaldiWriter(object):
    def __init__(self, feat_path, rank):
        os.makedirs(feat_path, exist_ok=True)
        lens_file = f"{feat_path}/feats.{rank}.len"
        ark_file = f"{feat_path}/feats.{rank}.ark"
        ark_scp_file = f"{feat_path}/feats.{rank}.scp"
        self.feat_path = kaldiio.WriteHelper(f"ark,scp:{ark_file},{ark_scp_file}", compression_method=1)
        self.leng_path = open(lens_file, "w", encoding="utf-8")

    def close(self):
        try:
            self.leng_path.close()
        except Exception as e:
            print("Error! close leng_path failed!!!")
            pass
        try:
            self.feat_path.close()
        except Exception as e:
            print("Error! close kaldi feat_path failed!!!")
            raise e

    def save_feats(self, feat, utt):
        self.feat_path(utt, feat)
        self.leng_path.write(f"{utt}\t{len(feat)}\n")

def read_npy_feat(feat_path, rank, percent, use_iterate=False):
    npy_file = f"{feat_path}/feats.{rank}.npy"
    lens_file = f"{feat_path}/feats.{rank}.len"
    with open(lens_file, "r") as f:
        utts, lengs = zip(*((parts[0], int(parts[1])) for line in f for parts in [line.strip().split("\t")]))
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if use_iterate:
        def iterate():
            feat = np.load(npy_file, mmap_mode="r")
            assert feat.shape[0] == (offsets[-1] + lengs[-1])
            for offset, leng in zip(offsets, lengs):
                yield feat[offset: offset + leng]

        return iterate, len(lengs), utts
    
    if percent < 0:
        return np.load(npy_file, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(npy_file, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(f"sampled {nsample} utterances, {len(sampled_feat)} frames ")
        return sampled_feat

def read_kaldi_feat(feat_path, rank, percent, use_iterate=False):
    ark_scp_file = f"{feat_path}/feats.{rank}.scp"  # refer to ark
    with open(ark_scp_file, "r") as f:
        utts, feats = zip(*((parts[0], parts[1]) for line in f for parts in [line.strip().split()]))
    
    if use_iterate:
        def iterate():
            with kaldiio.ReadHelper(f"scp:{ark_scp_file}") as reader:
                for key, feat in reader:
                    yield feat

        return iterate, len(feats), utts

    if percent < 0:
        selected_utts = utts
        nsample = len(feats)
    else:
        nsample = int(np.ceil(len(feats) * percent))
        selected_indices = np.random.choice(len(feats), nsample, replace=False)
        selected_utts = set([utts[i] for i in selected_indices])

    all_matrix_lens = 0
    all_matrix_dim = 0
    with kaldiio.ReadHelper(f"scp:{ark_scp_file}") as reader:
        for utt_id, feat in reader:
            if utt_id in selected_utts:
                all_matrix_lens += feat.shape[0]
                all_matrix_dim = feat.shape[1]

    sampled_feat = np.zeros([all_matrix_lens, all_matrix_dim], dtype=np.float32)
    idx = 0
    with kaldiio.ReadHelper(f"scp:{ark_scp_file}") as reader:
        for utt_id, feat in reader:
            if utt_id in selected_utts:
                sampled_feat[idx : idx + feat.shape[0]] = feat
                idx += feat.shape[0]

    logger.info(f"sampled {nsample} utterances, {len(sampled_feat)} frames ")
    return sampled_feat
