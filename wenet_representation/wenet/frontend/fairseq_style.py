import torch
import types
import math
from typing import Optional, Tuple, List
from wenet.frontend.utils import FeatureProcessor

class BaseFrontend(torch.nn.Module):
    def __init__(
        self,
        padding_mask: bool = True,
        multilayer_feature: bool = False,
        feature_selection: str = None,
        num_layer: int = 13,
        layer: List[int] = [-1],
        down_sample: int = 320,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.layers = sorted(layer)
        self.multilayer_feature = multilayer_feature
        if self.layers != [-1]:
            layer_selections = self.layers
        else:
            layer_selections = None
        
        if self.layers == [-1] and not self.multilayer_feature:
            print(" not using FeatureProcessor")
            self.feature_processor = None
        else:
            self.feature_processor = FeatureProcessor(num_layer, layer_selections=layer_selections)
            if weight is not None:
                self.feature_processor.reset_weight(weight)

        self.down_sample_rate = down_sample
        assert feature_selection in ['layers', 'layers_before_residual', None], print(f"wrong feature_selection: {feature_selection}")
        self.feature_selection = feature_selection
        self.padding_mask = padding_mask

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        raise NotImplementedError

    @staticmethod
    def cal_conv_outputs(input, down_sample_rate):
        if down_sample_rate == 2:
            feature_enc_layers = [(512, 3, 2)]
        elif down_sample_rate == 4:
            feature_enc_layers = [(512, 3, 2), (512, 3, 2)]
        elif down_sample_rate == 8:
            feature_enc_layers = [(512, 5, 2), (512, 3, 2), (512, 3, 2)]
        else:
            feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]
        
        for layer in feature_enc_layers:
            input = math.floor((input-layer[1]) / layer[2] + 1)
        return input

class Data2vec2Frontend(BaseFrontend):
    def __init__(
        self, 
        model_dir: str = './checkpoint_best.pt',
        finetune_model: bool = False,
        padding_mask: bool = True,
        multilayer_feature: bool = False, 
        feature_selection: str = None, 
        num_layer: int = 13, 
        layer: List[int] = [-1], 
        down_sample: int = 320,
        weight: torch.Tensor = None,
        user_dir: str = None,
        **kwargs,
    ):
        super().__init__(
            padding_mask,
            multilayer_feature, 
            feature_selection, 
            num_layer, 
            layer, 
            down_sample,
            weight
        )
        if model_dir != "":
            try:
                import fairseq
            except Exception as e:
                print('Error: FairSeq is not properly installed.')
                raise e
        if user_dir != "":
            fairseq_args = types.SimpleNamespace()
            fairseq_args.user_dir = user_dir
            from fairseq.utils import import_user_module
            import_user_module(fairseq_args)

        models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [model_dir],
            arg_overrides={"data": model_dir,
                            "layerdrop": 0.0,
                            "skip_ema": True,},
        )

        model = models[0]
        model = model.eval()
        self.model = model
        self.use_ft_model = finetune_model
        self.task = task
        self.cfg = saved_cfg

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ):
        device = input.device
        if self.padding_mask:
            padding_mask = ~torch.lt(
                torch.arange(max(input_lengths)).unsqueeze(0).to(device),
                input_lengths.unsqueeze(1).to(device),
            )
        else:
            padding_mask = None

        if self.use_ft_model:
            net_input = {'source': input, 'padding_mask':padding_mask}
            with torch.no_grad():
                res = self.model(**net_input)
            encoder_out = res['encoder_out']
            hidden_states = [h[0] for h in res['layer_results']]
        else:
            with torch.no_grad():
                res = self.model.extract_features(input, padding_mask=padding_mask)
            hidden_states = [h[0] for h in res['layer_results']]
            hidden_states.append(res['x'])

        assert max(self.layers) < len(hidden_states), print(f"layer: {self.layers}, hidden_states_len: {len(hidden_states)}")
        feats_lens = input_lengths.cpu().clone().apply_(lambda x: self.cal_conv_outputs(x, self.down_sample_rate)).to(device)

        assert max(feats_lens) == hidden_states[0].size(1), print(f"not equal: feats_lens: {feats_lens}, hidden_states shape: {hidden_states[0].shape}")
        hidden_states = [h[:, :max(feats_lens), :] for h in hidden_states]

        if self.layers == [-1]:
            if self.multilayer_feature:
                feats = self.feature_processor(hidden_states)
            else:
                feats = hidden_states[-1]
        else:
            selected_hidden_states = [hidden_states[i] for i in self.layers]
            if self.multilayer_feature:
                feats = self.feature_processor(selected_hidden_states)
            else:
                hidden_states_stack = torch.stack(selected_hidden_states)
                feats = torch.mean(hidden_states_stack, dim=0)

        return feats, feats_lens