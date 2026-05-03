from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def normalized_cumulative_trace(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cum = trace.cumsum(dim=1)
    denom = cum[:, -1:].detach() + eps
    return cum / denom

def stack_temporal_features(feat_list: List[torch.Tensor]) -> torch.Tensor:
    if len(feat_list) == 0:
        raise RuntimeError("Empty temporal feature list.")
    if len(feat_list) == 1:
        return feat_list[0].unsqueeze(1)
    return torch.stack(feat_list, dim=1)

def temporal_state_loss(
    pre_seq: torch.Tensor,
    post_seq: torch.Tensor,
    lambda_curve: float = 1.0,
    lambda_diff: float = 0.5,
    lambda_mid: float = 0.5,
) -> torch.Tensor:
    pre_trace = spike_energy_trace(pre_seq)
    post_trace = spike_energy_trace(post_seq)

    pre_cum = normalized_cumulative_trace(pre_trace)
    post_cum = normalized_cumulative_trace(post_trace)

    loss_curve = F.l1_loss(post_cum, pre_cum.detach())

    if pre_trace.size(1) > 1:
        pre_diff = pre_trace[:, 1:] - pre_trace[:, :-1]
        post_diff = post_trace[:, 1:] - post_trace[:, :-1]
        loss_diff = F.l1_loss(post_diff, pre_diff.detach())
    else:
        loss_diff = pre_trace.new_zeros(())

    return lambda_curve * loss_curve + lambda_diff * loss_diff


def compute_temporal_stability(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if trace.size(1) <= 1:
        return trace.new_tensor(1.0)

    step_diff = (trace[:, 1:] - trace[:, :-1]).abs().mean(dim=1)
    trace_mean = trace.mean(dim=1) + eps
    stability = torch.exp(-step_diff / trace_mean)
    return stability.mean()

def spike_energy_trace(feat_seq: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if feat_seq.dim() != 5:
        raise ValueError(f"spike_energy_trace expects [B,T,C,H,W], got {tuple(feat_seq.shape)}")
    return feat_seq.abs().mean(dim=(2, 3, 4)) + eps

def compute_burst_ratio(trace: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    peak = trace.max(dim=1).values
    mean = trace.mean(dim=1) + eps
    return (peak / mean).mean()


class TemporalChannelAdapter(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_steps: int,
        init_alpha: float = 1.0,
        fast_lr: float = 0.02,
        fast_decay: float = 0.01,
        theta_momentum: float = 0.95,
        rate_momentum: float = 0.95,
        homeo_weight: float = 0.30,
        fast_clamp: float = 0.10,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_steps = num_steps

        self.scale_c = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias_c = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.scale_t = nn.Parameter(torch.zeros(num_steps, num_channels, 1, 1))
        self.bias_t = nn.Parameter(torch.zeros(num_steps, num_channels, 1, 1))
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        self.fast_lr = fast_lr
        self.fast_decay = fast_decay
        self.theta_momentum = theta_momentum
        self.rate_momentum = rate_momentum
        self.homeo_weight = homeo_weight
        self.fast_clamp = fast_clamp

        self.register_buffer("fast_weight", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("theta", torch.ones(1, num_channels, 1, 1) * 0.10)
        self.register_buffer("rate_ema", torch.zeros(1, num_channels, 1, 1))

        self._seq_step = 0

    def begin_sequence(self):
        self._seq_step = 0

    def passive_decay(self):
        with torch.no_grad():
            self.fast_weight.mul_(1.0 - self.fast_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = min(self._seq_step, self.num_steps - 1)

        scale_t = self.scale_t[t].unsqueeze(0)
        bias_t = self.bias_t[t].unsqueeze(0)

        delta_slow = x * (self.scale_c + scale_t) + (self.bias_c + bias_t)
        delta_fast = x * self.fast_weight

        y = x + self.alpha * (delta_slow + delta_fast)

        self._seq_step += 1
        return y

    @torch.no_grad()
    def apply_fast_plasticity(self, pre_seq: torch.Tensor, post_seq: torch.Tensor, gate: torch.Tensor):
        if isinstance(gate, torch.Tensor):
            g = float(torch.clamp(gate.detach(), 0.0, 1.0).item())
        else:
            g = float(max(0.0, min(1.0, gate)))

        pre_mean = pre_seq.mean(dim=(0, 1, 3, 4), keepdim=True)
        pre_std = pre_seq.std(dim=(0, 1, 3, 4), keepdim=True, unbiased=False) + 1e-6
        post_mean = post_seq.mean(dim=(0, 1, 3, 4), keepdim=True)
        post_std = post_seq.std(dim=(0, 1, 3, 4), keepdim=True, unbiased=False) + 1e-6

        pre_norm = (pre_seq - pre_mean) / pre_std
        post_norm = (post_seq - post_mean) / post_std

        corr = (pre_norm * post_norm).mean(dim=(0, 1, 3, 4), keepdim=True).squeeze(1)

        rate = post_seq.abs().mean(dim=(0, 1, 3, 4), keepdim=True).squeeze(1)

        if self.rate_ema.abs().sum().item() < 1e-12:
            self.rate_ema.copy_(rate)

        self.theta.mul_(self.theta_momentum).add_((1.0 - self.theta_momentum) * rate.pow(2))
        self.rate_ema.mul_(self.rate_momentum).add_((1.0 - self.rate_momentum) * rate)

        hebb_term = self.fast_lr * g * corr * (rate - self.theta)
        homeo_term = - self.fast_lr * g * self.homeo_weight * (rate - self.rate_ema)

        self.fast_weight.mul_(1.0 - self.fast_decay)
        self.fast_weight.add_(hebb_term + homeo_term)
        self.fast_weight.clamp_(-self.fast_clamp, self.fast_clamp)


class PAR(nn.Module):
    def __init__(self, model, optimizer=None, steps: int = 1, episodic: bool = False, input_shape=(1, 3, 32, 32),
        target_layers=("pool3"),

        stability_thresh: float = 0.65,
        burst_tolerance: float = 1.80,
        adapt_gate_min: float = 0.35,
        gate_scale: float = 8.0,

        lambda_struct: float = 1.0,
        lambda_orth: float = 0.05,
        lambda_temporal: float = 0.60,

        fast_lr: float = 0.02,
        fast_decay: float = 0.01,
        theta_momentum: float = 0.95,
        rate_momentum: float = 0.95,
        homeo_weight: float = 0.30,
        fast_clamp: float = 0.10,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.target_layers = tuple(target_layers)

        self.stability_thresh = stability_thresh
        self.burst_tolerance = burst_tolerance
        self.adapt_gate_min = adapt_gate_min
        self.gate_scale = gate_scale

        self.lambda_struct = lambda_struct
        self.lambda_orth = lambda_orth
        self.lambda_temporal = lambda_temporal

        self.fast_lr = fast_lr
        self.fast_decay = fast_decay
        self.theta_momentum = theta_momentum
        self.rate_momentum = rate_momentum
        self.homeo_weight = homeo_weight
        self.fast_clamp = fast_clamp

        assert steps > 0, "PAR requires >= 1 adaptation step(s)."

        if hasattr(model, "num_steps"):
            self.num_steps = int(model.num_steps)
        else:
            raise ValueError("Backbone model must have attribute 'num_steps' for SNN temporal adaptation.")

        self.adapters = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        self._collect_features = False
        self._current_features: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._hook_handles = []

        self._build_heads(input_shape)
        self._register_adaptation_hooks()

        self.model_state = None
        self.optimizer_state = None
        if self.optimizer is not None:
            self.model_state, self.optimizer_state = copy_model_and_optimizer(self, self.optimizer)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self, self.optimizer)

    @torch.no_grad()
    def _build_heads(self, input_shape):
        module_dict = dict(self.model.named_modules())

        for name in self.target_layers:
            if name not in module_dict:
                raise ValueError(f"[PAR] target layer '{name}' not found in backbone.")

        channel_cache = {}
        tmp_handles = []

        def make_shape_hook(layer_name):
            def hook(module, inp, out):
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if not torch.is_tensor(out):
                    raise TypeError(f"[PAR] layer '{layer_name}' output is not a tensor.")
                channel_cache[layer_name] = out.shape[1]
            return hook

        for name in self.target_layers:
            h = module_dict[name].register_forward_hook(make_shape_hook(name))
            tmp_handles.append(h)

        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()

        dummy = torch.zeros(*input_shape, device=device)
        out = self.model(dummy)
        _ = out[0] if isinstance(out, (tuple, list)) else out

        for h in tmp_handles:
            h.remove()

        if was_training:
            self.model.train()

        for name in self.target_layers:
            c = channel_cache[name]
            self.adapters[name] = TemporalChannelAdapter(num_channels=c, num_steps=self.num_steps, fast_lr=self.fast_lr, fast_decay=self.fast_decay,
                theta_momentum=self.theta_momentum, rate_momentum=self.rate_momentum, homeo_weight=self.homeo_weight, fast_clamp=self.fast_clamp).to(device)

    def _register_adaptation_hooks(self):
        module_dict = dict(self.model.named_modules())

        def make_hook(layer_name):
            def hook(module, inp, out):
                if isinstance(out, (tuple, list)):
                    raise TypeError(f"[DTPA] layer '{layer_name}' output must be a tensor.")
                if not torch.is_tensor(out):
                    raise TypeError(f"[DTPA] layer '{layer_name}' output is not a tensor.")

                pre_feat = out.detach()
                post_feat = self.adapters[layer_name](out)

                if self._collect_features:
                    self._current_features[layer_name]["pre"].append(pre_feat)
                    self._current_features[layer_name]["post"].append(post_feat)

                return post_feat
            return hook

        for name in self.target_layers:
            h = module_dict[name].register_forward_hook(make_hook(name))
            self._hook_handles.append(h)

    def _begin_sequence(self):
        for m in self.adapters.values():
            m.begin_sequence()

    def _passive_decay_fast_weights(self):
        for m in self.adapters.values():
            m.passive_decay()

    @torch.no_grad()
    def _compute_reliability_gate(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        last_name = self.target_layers[-1]
        post_seq = stack_temporal_features(self._current_features[last_name]["post"]).detach()  # [B,T,C,H,W]
        trace = spike_energy_trace(post_seq)  # [B,T]

        stability = compute_temporal_stability(trace)

        gate_logit = stability - self.stability_thresh
        gate = torch.sigmoid(self.gate_scale * gate_logit)

        stats = {"stability": stability.detach(), "gate": gate.detach()}
        return gate, stats

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, dtpa_model: DTPA, optimizer):
    dtpa_model._collect_features = True
    dtpa_model._current_features = {name: {"pre": [], "post": []} for name in dtpa_model.target_layers}
    dtpa_model._begin_sequence()

    model_out = dtpa_model.model(x)
    outputs = model_out[0]

    dtpa_model._collect_features = False

    gate, gate_stats = dtpa_model._compute_reliability_gate(outputs)

    if gate.item() < dtpa_model.adapt_gate_min:
        optimizer.zero_grad(set_to_none=True)
        dtpa_model._passive_decay_fast_weights()
        return outputs.detach()

    total_loss = outputs.new_zeros(())
    per_layer_cache = {}
    loss_ent = softmax_entropy(outputs).mean()

    for name in dtpa_model.target_layers:
        feats = dtpa_model._current_features[name]
        pre_seq = stack_temporal_features(feats["pre"])    # [B,T,C,H,W]
        post_seq = stack_temporal_features(feats["post"])  # [B,T,C,H,W]

        loss_temp = dtpa_model.lambda_temporal * temporal_state_loss(pre_seq.detach(), post_seq)
        total_loss = total_loss + loss_temp
        per_layer_cache[name] = {"pre_seq": pre_seq.detach(), "post_seq": post_seq.detach()}

    total_loss = gate.detach() * (total_loss + loss_ent)

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        for name in dtpa_model.target_layers:
            dtpa_model.adapters[name].apply_fast_plasticity(per_layer_cache[name]["pre_seq"], per_layer_cache[name]["post_seq"], gate)

    return outputs.detach()


def collect_params(model):
    params = []
    names = []
    for n, p in model.adapters.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(f"adapters.{n}")

    for n, p in model.projectors.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(f"projectors.{n}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.requires_grad_(False)
    return model