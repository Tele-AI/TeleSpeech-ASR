import torch
import random

def spec_aug(speech, speech_lengths, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    assert isinstance(speech, torch.Tensor)
    bsz, Tmax, Fmax = speech.shape
    for b in range(bsz):
        max_frames = speech_lengths[b]
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            speech[b, start:end, :] = 0
        for i in range(num_f_mask):
            start = random.randint(0, Fmax - 1)
            length = random.randint(1, max_f)
            end = min(Fmax, start + length)
            speech[b, :, start:end] = 0
    return speech