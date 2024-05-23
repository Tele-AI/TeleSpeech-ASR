import os
import re
import sentencepiece as spm

data_dir = "/path/to/data"
feat_dir = "/path/to/feat"
bpemodel_dir = ""
use_bpe = False
vocab_size = 5000  # not used
feat_dim = 40

text_file = os.path.join(data_dir, "text")
datalist_file = os.path.join(data_dir, "data.list")
feat_scp = os.path.join(feat_dir, "mfcc.scp")
len_file = os.path.join(feat_dir, "feat2len.txt")
utt2ark = dict()
utt2len = dict()
utt2txt = dict()

with open(text_file, "r") as f:
    for line in f:
        line = line.strip().split()
        utt, text = line[0], " ".join(line[1:])
        utt2txt[utt] = text

with open(len_file, "r") as f:
    for line in f:
        utt, feat_len = line.strip().split()
        utt2len[utt] = int(feat_len)

sp = spm.SentencePieceProcessor()
if use_bpe:
    bpe_model = sp.load(bpemodel_dir)
    vocab_size = sp.get_piece_size()
with open(feat_scp, "r") as fin, open(datalist_file, "w") as fout:
    for line in fin:
        utt, ark = line.strip().split()
        txt, feat_len = utt2txt[utt], utt2len[utt]
        token_shape = len(txt.split())
        if use_bpe:
            token = " ".join(sp.EncodeAsPieces(txt))
        else:
            token = " ".join(re.sub(" ", "", txt))
        res_feat = f"utt:{utt}\tfeat:{ark}\tfeat_shape:{feat_len},{feat_dim}"
        res_text = f"text:{txt}\ttoken:{token}\ttokenid:[TOKENID]\ttoken_shape:{token_shape},{vocab_size}"
        res = f"{res_feat}\t{res_text}"
        print(res, file=fout)
