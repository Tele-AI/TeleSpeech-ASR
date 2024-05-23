# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import editdistance
from torch.nn.utils.rnn import pad_sequence


from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                th_accuracy, remove_duplicates_and_blank,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.frontend.spec_aug import spec_aug


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        feat_extractor = None,
        pre_encoder = None,
        spec_aug_conf : dict = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.feat_extractor = feat_extractor
        self.pre_encoder = pre_encoder
        self.spec_aug_conf = spec_aug_conf
        if self.spec_aug_conf:
            print("using spec_aug after feat extractor")

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 0.0 feature extractor from pretrained model
        if self.feat_extractor:
            speech, speech_lengths = self.feat_extractor(speech, speech_lengths)  # [B, T, Dim]  e.g. [B, T, 768]
        
        # 0.1 pre-encoder for extractor
        if self.pre_encoder:
            speech, speech_lengths = self.pre_encoder(speech, speech_lengths)  # [B, T, 80]
            
        # 0.2 spec_aug_after
        if self.spec_aug_conf:
            speech = spec_aug(speech, speech_lengths, **self.spec_aug_conf)

        # 1. Encoder
        if self.encoder:
            encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        else:
            encoder_out = speech
            encoder_out_lens = speech_lengths

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
            cer, err_char, sum_tok = self._calc_cer(encoder_out, encoder_out_lens, text,
                                text_lengths)

            cer = torch.tensor(float(cer))
            err_char = torch.tensor(err_char)
            sum_tok = torch.tensor(sum_tok)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att

        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc, "c_err": cer, "err_char": err_char, "sum_tok": sum_tok}

    def _calc_cer(self, encoder_out, encoder_out_lens, text, text_lengths):
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        c_err = 0
        for b in range(ctc_probs.size(0)):
            toks = ctc_probs[b, :, :].argmax(dim=-1).unique_consecutive()
            pred_units_arr = toks[(toks != 0) & (toks != -1)].tolist()
            txt = text[b, :]
            target_units_arr = txt[(txt != 0) & (txt != -1)].tolist()
            err_char = editdistance.eval(pred_units_arr, target_units_arr)
            c_err += err_char
        return "{:.2f}".format(100 * float(c_err) / sum(text_lengths)), c_err, sum(text_lengths).item()

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            # 0.0 feature extractor from pretrained model
            if self.feat_extractor:
                speech, speech_lengths = self.feat_extractor(speech, speech_lengths)

            # 0.1 pre-encoder for extractor
            if self.pre_encoder:
                speech, speech_lengths = self.pre_encoder(speech, speech_lengths)

            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            # Update cache to be consistent with new topk scores / hyps
            cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
            base_cache_index = (torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
            cache_index = base_cache_index + cache_index
            cache = [torch.index_select(c, dim=0, index=cache_index) for c in cache]
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        return hyps[0]

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score
