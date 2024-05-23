import logging
import os
import sys
import json
import re

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("change data.list for training")

def main(subset, lab_dir, datalist_dir, ori_type):
    km_labels = os.path.join(lab_dir, subset, "feat.km")
    datalist = os.path.join(datalist_dir, subset, "data.list")
    save_datalist = os.path.join(lab_dir, subset, "data.list.discrete")
    utt2toks = dict()
    with open(km_labels, "r") as fin:
        for line in fin:
            line = line.strip().split("\t")
            assert len(line) == 2
            assert line[0] not in utt2toks, print(f"repeated utt_id: {line[0]}")
            utt2toks[line[0]] = line[1]

    with open(datalist, "r") as fin, open(save_datalist, "w") as fout:
        for line in fin:
            if ori_type == "raw":
                data = json.loads(line)
                key, txt = data["key"], data["txt"]
            elif ori_type == "wenet_ark":
                data = line.strip().split("\t")
                if len(data) != 7:
                    logger.info(f"data {data} not regular, skip this line")
                    continue
                key, txt = re.sub("utt:", "", data[0]), re.sub("text:", "", data[3])
            toks = utt2toks[key]
            res = json.dumps(dict(key=key, token=toks, txt=txt), ensure_ascii=False)
            print(res, file=fout)
            
    logger.info(f"finished {km_labels} successfully")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str)
    parser.add_argument("--lab_dir", type=str)
    parser.add_argument("--datalist_dir", type=str)
    parser.add_argument(
        "--ori_type",
        type=str,
        default="raw",
        choices=["raw", "kaldi_ark", "wenet_ark"],
    )
    args = parser.parse_args()
    logging.info(str(args))

    main(**vars(args))