# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn

from wenet.pre_encoder.linear import LinearProjection
from wenet.frontend.fairseq_style import BaseFrontend, Data2vec2Frontend
from wenet.frontend.embed import EmbeddingFrontend

feat_extractor_choice = dict(
    base=BaseFrontend,
    d2v2=Data2vec2Frontend,
    embed=EmbeddingFrontend,
)


def init_model_with_feat_extractor(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    # feat_extracter
    frontend_conf = configs.get('frontend_conf', None)
    if frontend_conf:
        feat_extractor_type = frontend_conf.get('feat_extractor_type', None)
        feat_extractor_conf= frontend_conf.get('feat_extractor_conf', {})

        feat_extractor_class = feat_extractor_choice.get(feat_extractor_type, None)
        if feat_extractor_class is None:
            feat_extractor = None
        else:
            feat_extractor = feat_extractor_class(**feat_extractor_conf)

    # pre_encoder
    preencoder_conf = configs.get('preencoder_conf', None)
    if preencoder_conf:
        preencoder_conf = configs.get('preencoder_conf', None)
        input_size = preencoder_conf['input_size']
        output_size = preencoder_conf['output_size']
        dropout = preencoder_conf.get('dropout_rate', 0.0)
        
        pre_encoder = LinearProjection(input_size=input_size,
                                       output_size=output_size,
                                       dropout=dropout)
    else:
        output_size = frontend_conf['output_dim']
        pre_encoder = None

    spec_aug_after_conf = configs.get('spec_aug_after_conf', None)
    if spec_aug_after_conf:
        scale = spec_aug_after_conf.get('scale', False)
        if scale:
            down_sample = (2 if 'down_sample' not in feat_extractor_conf or 
                           feat_extractor_conf['down_sample'] == 320 
                           else feat_extractor_conf['down_sample'])
            print(f"scaling for spec_aug_after, down_sample rate is {down_sample}")
            spec_aug_after_conf['max_t'] = math.floor(spec_aug_after_conf['max_t'] / down_sample)
            spec_aug_after_conf['max_f'] = math.floor(spec_aug_after_conf['max_f'] * output_size / 80)
            del spec_aug_after_conf['scale']
        
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'transformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    elif encoder_type == 'none':
        encoder = None
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'none':
        decoder = None
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention
    model = ASRModel(vocab_size=vocab_size,
                    encoder=encoder,
                    decoder=decoder,
                    ctc=ctc,
                    feat_extractor=feat_extractor,
                    pre_encoder=pre_encoder,
                    spec_aug_conf=spec_aug_after_conf,
                    **configs['model_conf'])
    return model