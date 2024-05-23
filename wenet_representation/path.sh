export WENET_DIR=/path/to/wenet_representation/wenet
# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=/path/to/fairseq:/path/to/data2vec_dialect:$PYTHONPATH
