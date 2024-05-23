import logging
import os
import sys
import re
import argparse
import yaml
import torch
import kaldiio
import soundfile as sf
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from wenet.discrete_token.utils import KaldiWriter, NpyWriter
from wenet.utils.init_model import feat_extractor_choice

writer_choice = dict(
    kaldi_ark=KaldiWriter,
    npy=NpyWriter,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_pretrain_feature")

class BaseFeatureReader(object):
    def __init__(self):
        raise NotImplementedError  
      
    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav, len(wav)

    def read_ark(self, path):
        mat = kaldiio.load_mat(path)
        # with torch.no_grad():
        #     feat = torch.from_numpy(mat).float()        
        return mat, len(mat)

    def get_feats(self, path, input_type="raw", ref_len=None):
        raise NotImplementedError

    def normalize(self, data, input_type):
        if input_type == "raw":
            x = F.layer_norm(data, data.shape)
        elif input_type == "wenet_ark":
            assert data.dim() == 2, data.dim()
            m = data.mean(dim=0)
            std = data.std(dim=0)
            x = (data - m) / (std + 1e-5)            
        return x

    @staticmethod
    def cal_samples(down_sample_rate=320):
        if down_sample_rate == 2:
            feature_enc_layers = [(512, 3, 2)]
        elif down_sample_rate == 4:
            feature_enc_layers = [(512, 3, 2), (512, 3, 2)]
        elif down_sample_rate == 8:
            feature_enc_layers = [(512, 5, 2), (512, 3, 2), (512, 3, 2)]
        else:
            feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]
        
        jin = 0
        rin = 0
        for _, k, stride in feature_enc_layers:
            if rin == 0:
                rin = k
            rin = rin + (k - 1) * jin
            if jin == 0:
                jin = stride
            else:
                jin *= stride
        return rin


class PretrainFeatureReader(BaseFeatureReader):
    def __init__(
        self,
        model_type: str = "w2v2",
        model_dir: str = "./checkpoint_best.pt",
        finetune_model: bool = False,
        normalize_before: bool = False,
        multilayer_feature: bool = False,
        feature_selection: str = None,
        num_layer: int = 13,
        layer: List[int] = [-1],
        padding_mask: bool = True,
        down_sample_rate: int = 320,
        max_chunk: int = -1,
        weights: List = None,
        user_dir: str = None,
        **kwargs,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_class = feat_extractor_choice.get(model_type, None)
        if weights is not None:
            weights = torch.tensor(weights)
        else:
            weights = None
            
        self.model = model_class(
            model_dir=model_dir,
            finetune_model=finetune_model,
            padding_mask=padding_mask,
            multilayer_feature=multilayer_feature, 
            feature_selection=feature_selection, 
            num_layer=num_layer, 
            layer=layer, 
            down_sample=down_sample_rate,
            weight=weights,
            user_dir=user_dir,
        )
        self.model = self.model.to(self.device)
        self.normalize_before = normalize_before
        self.max_chunk = max_chunk
        self.min_length = self.cal_samples(down_sample_rate=down_sample_rate)
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f" min_length = {self.min_length}")
        logger.info(f" normalize_before = {self.normalize_before}")

    def get_feats(self, msg, input_type="raw", ref_len=None):
        if input_type == "raw":
            msg = msg.strip().split()
            assert len(msg) == 2
            utt, path = msg
            x, x_lens = self.read_audio(path, ref_len)
        elif input_type == "wenet_ark":
            msg = msg.strip().split("\t")
            assert len(msg) == 7, print(msg)
            utt = re.sub("utt:", "", msg[0])
            path = re.sub("feat:", "", msg[1])
            x, x_lens = self.read_ark(path)
            
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.normalize_before:
                x = self.normalize(x, input_type=input_type)
            x = x.unsqueeze(0).float().to(self.device)  # [T, *] -> [B, T, *]
            if x.size(1) < self.min_length:
                logger.info(f" wrong wav: {path}, min length is {self.min_length}, wav shape is {x.shape}")
                raise RuntimeError
            
            x_lens = torch.tensor([x_lens]).long()
            if self.max_chunk > 0:
                feat = []
                for start in range(0, x.size(1), self.max_chunk):
                    x_chunk = x[:, start: start + self.max_chunk]
                    x_chunk_lens = torch.tensor([x_chunk.size(1)]).long()
                    feat_chunk, feats_lens = self.model(x_chunk, x_chunk_lens)
                    feat.append(feat_chunk.cpu()) 
                feats = torch.cat(feat, 1)
            else:
                feats, feats_lens = self.model(x, x_lens)
        feats = feats.cpu().squeeze(0).numpy()
        return feats, utt

def dump_feature(root_dir, subset, feat_dir, config, rank, nshard, input_type, output_type):
    with open(config, "r") as fin:
        conf = yaml.load(fin, Loader=yaml.FullLoader)
    
    reader_conf = conf.get("reader_conf", None)
    reader = PretrainFeatureReader(**reader_conf)
    writer_class = writer_choice.get(output_type, None)
    writer = writer_class(feat_dir, rank)
    if nshard <=1:
        filename = "wav.scp" if input_type == "raw" else "data.list"
    else:
        filename = f"wav.{rank}.scp" if input_type == "raw" else f"data.{rank}.list"
    scp_set_dir = os.path.join(root_dir, subset, filename)
    logger.info(f"doing {scp_set_dir} dump_feature")
    with open(scp_set_dir, "r") as fin:
        for i, line in enumerate(fin):
            feat, utt = reader.get_feats(line, input_type=input_type)
            writer.save_feats(feat, utt)
    writer.close()

    logger.info("finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--feat_dir")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument(
        "--input_type",
        type=str,
        default="raw",
        choices=["raw", "kaldi_ark", "wenet_ark"],
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="npy",
        choices=["npy", "kaldi_ark"],
    )

    args = parser.parse_args()
    logger.info(args)

    dump_feature(**vars(args))