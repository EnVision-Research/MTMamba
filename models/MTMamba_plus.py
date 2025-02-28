import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from typing import Optional, Callable, Any
from collections import OrderedDict
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath, trunc_normal_

from .utils import PatchExpand, STMBlock

INTERPOLATE_MODE = 'bilinear'

class CSS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        ssm_ratio=2,
        dt_rank="auto",        
        # ======================
        dropout=0.,
        conv_bias=True,
        bias=False,
        dtype=None,
        # ======================
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        shared_ssm=False,
        softmax_version=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_cross = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K * inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True) # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True) # (K * D)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, x_cross: torch.Tensor):
        self.selective_scan = selective_scan_fn
        assert x.shape == x_cross.shape
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_hwwh_cross = torch.stack([x_cross.view(B, -1, L), torch.transpose(x_cross, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs_cross = torch.cat([x_hwwh_cross, torch.flip(x_hwwh_cross, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_cross, self.x_proj_weight)
        del x_cross, xs_cross, x_hwwh_cross

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y
    
    forward_core = forward_corev0

    def forward(self, x: torch.Tensor, x_cross: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x_cross = self.in_proj_cross(x_cross)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x, x_cross.permute(0,3,1,2))
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SCTM(nn.Module):
    def __init__(
        self,
        tasks,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio=2.0,
        shared_ssm=False,
        softmax_version=False,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.tasks = tasks
        self.norm_share = norm_layer(hidden_dim*len(tasks))
        self.conv_share = nn.Sequential(nn.Conv2d(hidden_dim*len(tasks), hidden_dim, 1),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))

        self.norm = nn.ModuleDict()
        self.op = nn.ModuleDict()
        for tname in self.tasks:
            self.norm[tname] = norm_layer(hidden_dim)
            self.op[tname] = CSS2D(
                d_model=hidden_dim, 
                dropout=attn_drop_rate, 
                d_state=d_state, 
                ssm_ratio=ssm_ratio, 
                dt_rank=dt_rank,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                **kwargs
            )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: dict):
        x_share = torch.cat([input[t] for t in self.tasks], dim=-1) 
        x_share = self.conv_share(self.norm_share(x_share).permute(0,3,1,2)).permute(0,2,3,1)
        out = {}
        for t in self.tasks:
            x_t = input[t]
            x = x_t + self.drop_path(self.op[t](self.norm[t](x_t), x_share))
            out[t] = x
        return out

class MTMamba_plus(nn.Module):
    def __init__(self, p, backbone, d_state=16, dt_rank="auto", ssm_ratio=2, mlp_ratio=0):
        super().__init__()
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.feature_channel = backbone.num_features
        self.img_size = p.IMAGE_ORI_SIZE

        each_stage_depth = 3
        stage_num = len(self.feature_channel) - 1

        dpr = [x.item() for x in torch.linspace(0.2, 0, stage_num*each_stage_depth)]

        self.expand_layers = nn.ModuleDict()
        self.concat_layers = nn.ModuleDict()
        self.block_stm = nn.ModuleDict()
        self.final_project = nn.ModuleDict()
        self.final_expand = nn.ModuleDict()
        for t in self.tasks:
            for stage in range(len(self.feature_channel) - 1):
                current_channel = self.feature_channel[::-1][stage]
                skip_channel = self.feature_channel[::-1][stage+1]

                self.expand_layers[f'{t}_{stage}'] = PatchExpand(input_resolution=None, 
                                                                dim=current_channel, 
                                                                dim_scale=2, 
                                                                norm_layer=nn.LayerNorm)
                self.concat_layers[f'{t}_{stage}'] = nn.Conv2d(2*skip_channel, skip_channel, 1)

                stm_layer = [STMBlock(hidden_dim=skip_channel,
                                drop_path=dpr[each_stage_depth*(stage)+stm_idx],
                                norm_layer=nn.LayerNorm,
                                ssm_ratio=ssm_ratio,
                                d_state=d_state,
                                mlp_ratio=mlp_ratio,
                                dt_rank=dt_rank) for stm_idx in range(2)]

                self.block_stm[f'{t}_{stage}'] = nn.Sequential(*stm_layer)

            self.final_expand[t] = nn.Sequential(
                nn.Conv2d(self.feature_channel[0], 96, 3, padding=1), 
                nn.SyncBatchNorm(96), 
                nn.ReLU(True)
            )
            trunc_normal_(self.final_expand[t][0].weight, std=0.02)

            self.final_project[t] = nn.Conv2d(96, p.TASKS.NUM_OUTPUT[t], 1)

        self.block_ctm = nn.ModuleDict()
        for stage in range(len(self.feature_channel) - 1):
            skip_channel = self.feature_channel[::-1][stage+1]

            ctm_layer = [SCTM(tasks=self.tasks,
                            hidden_dim=skip_channel,
                            drop_path=dpr[each_stage_depth*(stage)+2],
                            norm_layer=nn.LayerNorm,
                            ssm_ratio=ssm_ratio,
                            d_state=d_state,
                            mlp_ratio=mlp_ratio,
                            dt_rank=dt_rank)]
            self.block_ctm[f'{stage}'] = nn.Sequential(*ctm_layer)

    def _forward_expand(self, x_dict, selected_fea: list, stage: int) -> dict:
        if stage == 0:
            x_dict = {t: selected_fea[-1] for t in self.tasks}

        skip = selected_fea[::-1][stage+1]
        out = {}
        for t in self.tasks:
            x = self.expand_layers[f'{t}_{stage}'](x_dict[t])
            x = torch.cat((x.permute(0,3,1,2), skip), 1)
            x = self.concat_layers[f'{t}_{stage}'](x)
            x = x.permute(0,2,3,1)
            out[t] = x
        return out # B,H,W,C

    def _forward_block_stm(self, x_dict: dict, stage: int) -> dict:
        out = {}
        for t in self.tasks:
            out[t] = self.block_stm[f'{t}_{stage}'](x_dict[t])
        return out

    def _forward_block_ctm(self, x_dict: dict, stage: int) -> dict:
        return self.block_ctm[f'{stage}'](x_dict)


    def forward(self, x):
        # img_size = x.size()[-2:]

        # Backbone 
        selected_fea = self.backbone(x)

        x_dict = None
        for stage in range(len(self.feature_channel) - 1):
            x_dict = self._forward_expand(x_dict, selected_fea, stage)
            x_dict = self._forward_block_stm(x_dict, stage)
            x_dict = self._forward_block_ctm(x_dict, stage)
            x_dict = {t: xx.permute(0,3,1,2) for t, xx in x_dict.items()}

        out = {}
        for t in self.tasks:
            z = self.final_expand[t](x_dict[t])
            z = self.final_project[t](z)
            out[t] = F.interpolate(z, self.img_size, mode=INTERPOLATE_MODE)

        return out
