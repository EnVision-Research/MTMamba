import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PatchExpand, FinalPatchExpand_X4, STMBlock
from .CTM import CTMBlock

INTERPOLATE_MODE = 'bilinear'

class MTMamba(nn.Module):
    def __init__(self, p, backbone, d_state=16, dt_rank="auto", ssm_ratio=2, mlp_ratio=0):
        super().__init__()
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.feature_channel = backbone.num_features

        total_depth = 3
        dpr = [x.item() for x in torch.linspace(0.2, 0, (len(self.feature_channel)-1)*total_depth)]

        self.expand_layers = nn.ModuleDict()
        self.concat_layers = nn.ModuleDict()
        self.block_1 = nn.ModuleDict()
        self.block_2 = nn.ModuleDict()
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
                self.concat_layers[f'{t}_{stage}'] = nn.Linear(2*skip_channel, skip_channel)

                self.block_1[f'{t}_{stage}'] = STMBlock(hidden_dim=skip_channel,
                                                drop_path=dpr[total_depth*(stage)+0],
                                                norm_layer=nn.LayerNorm,
                                                ssm_ratio=ssm_ratio,
                                                d_state=d_state,
                                                mlp_ratio=mlp_ratio,
                                                dt_rank=dt_rank)
                self.block_2[f'{t}_{stage}'] = STMBlock(hidden_dim=skip_channel,
                                                drop_path=dpr[total_depth*(stage)+1],
                                                norm_layer=nn.LayerNorm,
                                                ssm_ratio=ssm_ratio,
                                                d_state=d_state,
                                                mlp_ratio=mlp_ratio,
                                                dt_rank=dt_rank)

            self.final_expand[t] = FinalPatchExpand_X4(
                                        input_resolution=None,
                                        dim=self.feature_channel[0],
                                        dim_scale=4,
                                        norm_layer=nn.LayerNorm,
                                    )
            self.final_project[t] = nn.Conv2d(self.feature_channel[0], p.TASKS.NUM_OUTPUT[t], 1, 1, 0, bias=True)

        self.block_3 = nn.ModuleDict()
        for stage in range(len(self.feature_channel) - 1):
            skip_channel = self.feature_channel[::-1][stage+1]
            self.block_3[f'{stage}'] = CTMBlock(tasks=self.tasks,
                                                hidden_dim=skip_channel,
                                                drop_path=dpr[total_depth*(stage)+2],
                                                norm_layer=nn.LayerNorm,
                                                ssm_ratio=ssm_ratio,
                                                d_state=d_state,
                                                mlp_ratio=mlp_ratio,
                                                dt_rank=dt_rank)

    def _forward_expand(self, x_dict, selected_fea: list, stage: int) -> dict:
        if stage == 0:
            x_dict = {t: selected_fea[-1] for t in self.tasks}

        skip = selected_fea[::-1][stage+1]
        out = {}
        for t in self.tasks:
            x = self.expand_layers[f'{t}_{stage}'](x_dict[t])
            x = torch.cat((x, skip.permute(0,2,3,1)), -1)
            x = self.concat_layers[f'{t}_{stage}'](x)
            out[t] = x
        return out # B,H,W,C

    def _forward_block1(self, x_dict: dict, stage: int) -> dict:
        out = {}
        for t in self.tasks:
            out[t] = self.block_1[f'{t}_{stage}'](x_dict[t])
        return out

    def _forward_block2(self, x_dict: dict, stage: int) -> dict:
        out = {}
        for t in self.tasks:
            out[t] = self.block_2[f'{t}_{stage}'](x_dict[t])
        return out

    def _forward_block3(self, x_dict: dict, stage: int) -> dict:
        return self.block_3[f'{stage}'](x_dict)

    def forward(self, x):
        img_size = x.size()[-2:]

        # Backbone 
        selected_fea = self.backbone(x)

        x_dict = None
        for stage in range(len(self.feature_channel) - 1):
            x_dict = self._forward_expand(x_dict, selected_fea, stage)
            x_dict = self._forward_block1(x_dict, stage)
            x_dict = self._forward_block2(x_dict, stage)
            x_dict = self._forward_block3(x_dict, stage)
            x_dict = {t: xx.permute(0,3,1,2) for t, xx in x_dict.items()}

        out = {}
        for t in self.tasks:
            z = self.final_expand[t](x_dict[t])
            z = self.final_project[t](z.permute(0,3,1,2))
            out[t] = F.interpolate(z, img_size, mode=INTERPOLATE_MODE)

        return out