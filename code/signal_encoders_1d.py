"""Parameter-matched 1-D sequence encoders for coherent radar responses.

The legacy controlled families share a strided convolutional tokenizer.  The
``Direct*`` families instead preserve every input bin and use a per-bin linear
embedding (except the native 1-D CNN control).  All direct non-CNN families
share the same coordinate embedding, global mean pooling, and output
projection so that the sequence mixer is the controlled factor.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from net.transolver3_1d import Transolver3Block1D, Transolver3SignalEncoder1D


class ConvTokenStem1D(nn.Module):
    """Match the tokenizer used by the Transolver-3 radar encoder."""

    def __init__(self, input_channels: int, base_channels: int):
        super().__init__()
        self.hidden = int(base_channels) * 4
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, self.hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.coordinate_embedding = nn.Sequential(nn.Linear(1, self.hidden), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(x).transpose(1, 2).contiguous()
        coordinate = torch.linspace(-1.0, 1.0, tokens.shape[1], device=x.device, dtype=x.dtype)
        return tokens + self.coordinate_embedding(coordinate.view(1, -1, 1))


class _PooledSignalEncoder1D(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, base_channels: int):
        super().__init__()
        self.tokenizer = ConvTokenStem1D(input_channels, base_channels)
        self.hidden = self.tokenizer.hidden
        self.final_norm = nn.LayerNorm(self.hidden)
        self.output = nn.Linear(self.hidden, output_dim)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.mix(self.tokenizer(x))
        return self.output(self.final_norm(tokens).mean(dim=1))


class TransformerSignalEncoder1D(_PooledSignalEncoder1D):
    """Pre-norm Transformer with a 3x feed-forward expansion."""

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        block = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=heads,
            dim_feedforward=self.hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=max(1, int(layers)), enable_nested_tensor=False)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.blocks(tokens)


class XLSTMSignalEncoder1D(_PooledSignalEncoder1D):
    """Official mLSTM-only xLSTM after the shared strided CNN tokenizer."""

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        max_context_length: int = 32,
        proj_factor: float = 2.05,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        try:
            from xlstm import (
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except Exception as exc:  # pragma: no cover - optional official package
            raise RuntimeError(
                "CNN-tokenized xLSTM requires the official NX-AI xlstm package; "
                "install xlstm==2.0.5 and use its mLSTM-only PyTorch backend"
            ) from exc

        if self.hidden % heads != 0:
            raise ValueError(f"xLSTM hidden width {self.hidden} must be divisible by {heads} heads")
        config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=heads,
                    proj_factor=proj_factor,
                )
            ),
            context_length=int(max_context_length),
            num_blocks=max(1, int(layers)),
            embedding_dim=self.hidden,
            slstm_at=[],
            dropout=dropout,
        )
        self.blocks = xLSTMBlockStack(config)
        self.implementation = "NX-AI/xlstm mLSTM-only parallel_stabilized_simple"
        self.proj_factor = float(proj_factor)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.blocks(tokens)


class ResidualConvBlock1D(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + self.dropout(x)


class ConvSignalEncoder1D(_PooledSignalEncoder1D):
    """Pure 1-D residual CNN mixer after the shared tokenizer."""

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        self.blocks = nn.Sequential(*[ResidualConvBlock1D(self.hidden, dropout) for _ in range(max(1, int(layers)))])
        bottleneck = max(64, self.hidden * 2 // 3)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(self.hidden, bottleneck, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(bottleneck, self.hidden, kernel_size=1),
        )

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.blocks(tokens.transpose(1, 2))
        features = features + self.channel_mlp(features)
        return features.transpose(1, 2).contiguous()


class ConvTransformerHybridSignalEncoder1D(_PooledSignalEncoder1D):
    """Residual CNN augmented by one global self-attention layer.

    ``placement=pre`` applies global attention to the tokenizer output before
    the local residual CNN mixer.  ``placement=post`` applies the identical
    attention layer after the CNN and channel MLP.  The tokenizer, CNN depth,
    pooling, and output projection are otherwise identical to
    :class:`ConvSignalEncoder1D`.
    """

    def __init__(
        self,
        placement: str,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        if placement not in {"pre", "post"}:
            raise ValueError(f"unsupported CNN-Transformer placement: {placement}")
        self.placement = placement
        self.blocks = nn.Sequential(*[ResidualConvBlock1D(self.hidden, dropout) for _ in range(max(1, int(layers)))])
        bottleneck = max(64, self.hidden * 2 // 3)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(self.hidden, bottleneck, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(bottleneck, self.hidden, kernel_size=1),
        )
        self.global_attention = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=heads,
            dim_feedforward=self.hidden * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def _mix_cnn(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.blocks(tokens.transpose(1, 2))
        features = features + self.channel_mlp(features)
        return features.transpose(1, 2).contiguous()

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.placement == "pre":
            tokens = self.global_attention(tokens)
        tokens = self._mix_cnn(tokens)
        if self.placement == "post":
            tokens = self.global_attention(tokens)
        return tokens


class RecurrentSignalEncoder1D(_PooledSignalEncoder1D):
    """Bidirectional vanilla-RNN or LSTM sequence mixer."""

    def __init__(
        self,
        cell: str,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        depth = max(1, int(layers))
        if cell == "rnn":
            recurrent_hidden = int(round(self.hidden * 13.0 / 12.0))
            recurrent_cls = nn.RNN
            kwargs = {"nonlinearity": "tanh"}
        elif cell == "lstm":
            recurrent_hidden = max(1, self.hidden * 2 // 3)
            recurrent_cls = nn.LSTM
            kwargs = {}
        else:
            raise ValueError(f"unsupported recurrent cell: {cell}")
        self.recurrent = recurrent_cls(
            input_size=self.hidden,
            hidden_size=recurrent_hidden,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
            **kwargs,
        )
        self.recurrent_projection = nn.Linear(recurrent_hidden * 2, self.hidden)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        output, _state = self.recurrent(tokens)
        return self.recurrent_projection(output)


class SelectiveStateSpaceBlock1D(nn.Module):
    """A transparent Mamba-style selective SSM reference block.

    This implements the input-dependent diagonal state-space recurrence in
    ordinary PyTorch.  It intentionally avoids the unavailable mamba-ssm CUDA
    extension, so it is suitable for correctness-oriented P40 comparisons but
    does not claim the latency of an official fused Mamba kernel.
    """

    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.inner = int(dim) * int(expand)
        self.state_dim = int(state_dim)
        self.dt_rank = max(1, math.ceil(dim / 16))
        self.norm = nn.LayerNorm(dim)
        self.in_projection = nn.Linear(dim, self.inner * 2)
        self.depthwise_conv = nn.Conv1d(
            self.inner,
            self.inner,
            kernel_size=3,
            padding=1,
            groups=self.inner,
        )
        self.x_projection = nn.Linear(self.inner, self.dt_rank + 2 * self.state_dim, bias=False)
        self.dt_projection = nn.Linear(self.dt_rank, self.inner)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, self.state_dim + 1, dtype=torch.float32)).repeat(self.inner, 1))
        self.skip = nn.Parameter(torch.ones(self.inner))
        self.out_projection = nn.Linear(self.inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u, gate = self.in_projection(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2))
        projected = self.x_projection(u)
        dt_raw, b_value, c_value = torch.split(projected, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_projection(dt_raw))
        a_value = -torch.exp(self.a_log.float()).to(dtype=x.dtype)
        state = x.new_zeros(x.shape[0], self.inner, self.state_dim)
        outputs = []
        for token_index in range(x.shape[1]):
            dt = delta[:, token_index]
            d_a = torch.exp(dt.unsqueeze(-1) * a_value.unsqueeze(0))
            d_b = dt.unsqueeze(-1) * b_value[:, token_index].unsqueeze(1)
            state = state * d_a + u[:, token_index].unsqueeze(-1) * d_b
            y = (state * c_value[:, token_index].unsqueeze(1)).sum(dim=-1)
            y = y + self.skip.to(dtype=x.dtype).unsqueeze(0) * u[:, token_index]
            outputs.append(y)
        y = torch.stack(outputs, dim=1) * F.silu(gate)
        return residual + self.dropout(self.out_projection(y))


class MambaStyleSignalEncoder1D(_PooledSignalEncoder1D):
    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 384,
        base_channels: int = 96,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(input_channels, output_dim, base_channels)
        self.blocks = nn.ModuleList(
            [SelectiveStateSpaceBlock1D(self.hidden, state_dim=16, expand=2, dropout=dropout) for _ in range(max(1, int(layers)))]
        )

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class FlattenMLPSignalEncoder1D(nn.Module):
    """Protocol-matched wideband DNN baseline without convolution or attention."""

    def __init__(
        self,
        input_channels: int = 6,
        output_dim: int = 384,
        hidden: int = 1024,
        layers: int = 3,
        dropout: float = 0.1,
        expected_length: int = 91,
    ):
        super().__init__()
        self.expected_length = int(expected_length)
        blocks: list[nn.Module] = [
            nn.Linear(int(input_channels) * self.expected_length, int(hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(layers) - 1)):
            blocks.extend([nn.Linear(int(hidden), int(hidden)), nn.GELU(), nn.Dropout(dropout)])
        blocks.extend([nn.LayerNorm(int(hidden)), nn.Linear(int(hidden), int(output_dim))])
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.expected_length:
            x = F.adaptive_avg_pool1d(x, self.expected_length)
        return self.network(x.reshape(x.shape[0], -1))


def build_signal_encoder_1d(
    family: str,
    input_channels: int,
    output_dim: int,
    base_channels: int,
    layers: int,
    heads: int,
    dropout: float,
) -> nn.Module:
    if family == "transolver3":
        return Transolver3SignalEncoder1D(
            input_channels=input_channels,
            output_dim=output_dim,
            base_channels=base_channels,
            layers=layers,
            heads=heads,
            slices=32,
            dropout=dropout,
        )
    if family == "transformer":
        return TransformerSignalEncoder1D(input_channels, output_dim, base_channels, layers, heads, dropout)
    if family == "xlstm":
        return XLSTMSignalEncoder1D(input_channels, output_dim, base_channels, layers, heads, dropout)
    if family == "cnn":
        return ConvSignalEncoder1D(input_channels, output_dim, base_channels, layers, dropout)
    if family == "cnn_transformer_pre":
        return ConvTransformerHybridSignalEncoder1D(
            "pre", input_channels, output_dim, base_channels, layers, heads, dropout
        )
    if family == "cnn_transformer_post":
        return ConvTransformerHybridSignalEncoder1D(
            "post", input_channels, output_dim, base_channels, layers, heads, dropout
        )
    if family in {"rnn", "lstm"}:
        return RecurrentSignalEncoder1D(family, input_channels, output_dim, base_channels, layers, dropout)
    if family == "mamba":
        return MambaStyleSignalEncoder1D(input_channels, output_dim, base_channels, layers, dropout)
    if family == "mlp":
        return FlattenMLPSignalEncoder1D(input_channels, output_dim, hidden=1024, layers=layers, dropout=dropout)
    raise ValueError(f"unsupported 1-D signal encoder family: {family}")


class DirectTokenEmbedding1D(nn.Module):
    """Embed each original frequency bin independently; no convolution."""

    def __init__(self, input_channels: int, hidden: int):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.coordinate_embedding = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_mlp(x.transpose(1, 2).contiguous())
        coordinate = torch.linspace(-1.0, 1.0, tokens.shape[1], device=x.device, dtype=x.dtype)
        return tokens + self.coordinate_embedding(coordinate.view(1, -1, 1))


class _DirectPooledSignalEncoder1D(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.embedding = DirectTokenEmbedding1D(input_channels, self.hidden)
        self.final_norm = nn.LayerNorm(self.hidden)
        self.output = nn.Linear(self.hidden, output_dim)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.mix(self.embedding(x))
        return self.output(self.final_norm(tokens).mean(dim=1))


class DirectTransolver3SignalEncoder1D(_DirectPooledSignalEncoder1D):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        slices: int,
        dropout: float,
    ):
        super().__init__(input_channels, output_dim, hidden)
        self.blocks = nn.ModuleList(
            [Transolver3Block1D(hidden, heads=heads, slices=slices, dropout=dropout) for _ in range(max(1, int(layers)))]
        )
        adapter_hidden = max(64, hidden * 7 // 12)
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, hidden),
        )

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens + self.adapter(tokens)


class DirectTransformerSignalEncoder1D(_DirectPooledSignalEncoder1D):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__(input_channels, output_dim, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 13 // 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=max(1, int(layers)), enable_nested_tensor=False)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.blocks(tokens)


class DirectXLSTMSignalEncoder1D(_DirectPooledSignalEncoder1D):
    """Official NX-AI mLSTM-only xLSTM stack over all original frequency bins.

    The input tokenizer remains the shared per-bin linear embedding.  The
    causal convolution inside each mLSTM layer is an intrinsic part of the
    published xLSTM block, rather than a separate CNN tokenizer.  We use the
    official PyTorch ``parallel_stabilized_simple`` mLSTM backend because the
    P40 (compute capability 6.1) cannot run the official sLSTM CUDA kernel.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        context_length: int = 91,
        proj_factor: float = 2.05,
    ):
        super().__init__(input_channels, output_dim, hidden)
        try:
            from xlstm import (
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except Exception as exc:  # pragma: no cover - depends on the optional official package
            raise RuntimeError(
                "direct xLSTM requires the official NX-AI xlstm package; "
                "install xlstm==2.0.5 and use its mLSTM-only PyTorch backend"
            ) from exc

        if hidden % heads != 0:
            raise ValueError(f"xLSTM hidden width {hidden} must be divisible by {heads} heads")
        config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=heads,
                    proj_factor=proj_factor,
                )
            ),
            context_length=int(context_length),
            num_blocks=max(1, int(layers)),
            embedding_dim=hidden,
            slstm_at=[],
            dropout=dropout,
        )
        self.blocks = xLSTMBlockStack(config)
        self.implementation = "NX-AI/xlstm mLSTM-only parallel_stabilized_simple"
        self.context_length = int(context_length)
        self.proj_factor = float(proj_factor)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.context_length:
            raise ValueError(
                f"direct frequency xLSTM expects {self.context_length} bins, got {tokens.shape[1]}"
            )
        return self.blocks(tokens)


class DirectMambaStyleSignalEncoder1D(_DirectPooledSignalEncoder1D):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__(input_channels, output_dim, hidden)
        self.blocks = nn.ModuleList(
            [SelectiveStateSpaceBlock1D(hidden, state_dim=16, expand=2, dropout=dropout) for _ in range(max(1, int(layers)))]
        )
        adapter_hidden = max(64, hidden * 5 // 6)
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, adapter_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, hidden),
        )

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens + self.adapter(tokens)


class DirectRecurrentSignalEncoder1D(_DirectPooledSignalEncoder1D):
    def __init__(
        self,
        cell: str,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__(input_channels, output_dim, hidden)
        depth = max(1, int(layers))
        if cell == "rnn":
            recurrent_hidden = int(round(hidden * 13.0 / 12.0))
            recurrent_cls = nn.RNN
            kwargs = {"nonlinearity": "tanh"}
        elif cell == "lstm":
            recurrent_hidden = max(1, hidden * 2 // 3)
            recurrent_cls = nn.LSTM
            kwargs = {}
        else:
            raise ValueError(f"unsupported recurrent cell: {cell}")
        self.recurrent = recurrent_cls(
            input_size=hidden,
            hidden_size=recurrent_hidden,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
            **kwargs,
        )
        self.recurrent_projection = nn.Linear(recurrent_hidden * 2, hidden)

    def mix(self, tokens: torch.Tensor) -> torch.Tensor:
        output, _state = self.recurrent(tokens)
        return self.recurrent_projection(output)


class PointwiseConvMLP1D(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DirectConvSignalEncoder1D(nn.Module):
    """Native full-resolution 1-D CNN; no separate shared tokenizer."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock1D(hidden, dropout) for _ in range(max(1, int(layers)))])
        self.channel_mixers = nn.Sequential(PointwiseConvMLP1D(hidden, dropout), PointwiseConvMLP1D(hidden, dropout))
        self.final_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.channel_mixers(self.blocks(self.input_conv(x))).transpose(1, 2).contiguous()
        return self.output(self.final_norm(features).mean(dim=1))


def build_direct_signal_encoder_1d(
    family: str,
    input_channels: int,
    output_dim: int,
    hidden: int,
    layers: int,
    heads: int,
    slices: int,
    dropout: float,
) -> nn.Module:
    if family == "transolver3":
        return DirectTransolver3SignalEncoder1D(input_channels, output_dim, hidden, layers, heads, slices, dropout)
    if family == "transformer":
        return DirectTransformerSignalEncoder1D(input_channels, output_dim, hidden, layers, heads, dropout)
    if family == "xlstm":
        return DirectXLSTMSignalEncoder1D(input_channels, output_dim, hidden, layers, heads, dropout)
    if family == "mamba":
        return DirectMambaStyleSignalEncoder1D(input_channels, output_dim, hidden, layers, dropout)
    if family in {"rnn", "lstm"}:
        return DirectRecurrentSignalEncoder1D(family, input_channels, output_dim, hidden, layers, dropout)
    if family == "cnn":
        return DirectConvSignalEncoder1D(input_channels, output_dim, hidden, layers, dropout)
    if family == "mlpcnn":
        return DirectMlpConvSignalEncoder1D(input_channels, output_dim, hidden, layers, dropout)
    if family == "mlptx":
        return DirectMlpTransformerSignalEncoder1D(input_channels, output_dim, hidden, layers, heads, dropout)
    raise ValueError(f"unsupported direct 1-D signal encoder family: {family}")


class DirectMlpConvSignalEncoder1D(nn.Module):
    """Global flattened MLP branch fused with a native full-resolution 1-D CNN.

    The MLP branch mirrors the strong global wideband map (nontokenized
    flatten over the full frequency view), while the CNN branch retains local
    spectral structure. The two 384-dim branches are added before the shared
    mean-pooled output projection.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        seq_len = 91
        self.mlp = FlattenMLPSignalEncoder1D(
            input_channels=input_channels,
            output_dim=output_dim,
            hidden=max(512, hidden),
            layers=3,
            dropout=dropout,
            expected_length=seq_len,
        )
        self.cnn = DirectConvSignalEncoder1D(
            input_channels=input_channels,
            output_dim=output_dim,
            hidden=hidden,
            layers=max(1, int(layers)),
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_feat = self.mlp(x)
        cnn_feat = self.cnn(x)
        return self.final_norm(mlp_feat + cnn_feat)


class DirectMlpTransformerSignalEncoder1D(nn.Module):
    """Global flattened MLP branch fused with a direct Transformer sequence branch."""

    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        seq_len = 91
        self.mlp = FlattenMLPSignalEncoder1D(
            input_channels=input_channels,
            output_dim=output_dim,
            hidden=max(1024, hidden),
            layers=3,
            dropout=dropout,
            expected_length=seq_len,
        )
        self.transformer = DirectTransformerSignalEncoder1D(
            input_channels=input_channels,
            output_dim=output_dim,
            hidden=hidden,
            layers=max(1, int(layers)),
            heads=heads,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_feat = self.mlp(x)
        tx_feat = self.transformer(x)
        return self.final_norm(mlp_feat + tx_feat)
