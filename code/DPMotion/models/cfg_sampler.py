import torch
import torch.nn as nn
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) * out_uncond
        return cfg_out


class CFGAdaptiveModel(nn.Module):
    """
    自适应CFG模型：根据输入维度自动选择调用forward或single_forward
    
    - 如果输入D = 2 * input_feats，调用model.forward（Group交互，两人拼接）
    - 如果输入D = input_feats，调用model.single_forward（Person交互，单人）
    """
    def __init__(self, model, cfg_scale, input_feats):
        super().__init__()
        self.model = model
        self.s = cfg_scale
        self.input_feats = input_feats
    
    def forward(self, x, timesteps, cond=None, mask=None, b=None):
        B, T, D = x.shape
        
        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)
        
        # 根据输入维度选择调用哪个函数
        if D == 2 * self.input_feats:
            # Group交互：两人拼接，调用forward
            out = self.model.forward(x_combined, timesteps_combined, cond=cond, mask=mask)
        elif D == self.input_feats:
            # Person交互：单人，调用single_forward
            if b is not None:
                b_combined = torch.cat([b, b], dim=0)
            else:
                b_combined = None
            out = self.model.single_forward(x_combined, timesteps_combined, cond=cond, mask=mask, b=b_combined)
        else:
            raise ValueError(f"Unexpected input dimension D={D}, expected {self.input_feats} or {2*self.input_feats}")
        
        out_cond = out[:B]
        out_uncond = out[B:]
        
        cfg_out = self.s * out_cond + (1-self.s) * out_uncond
        return cfg_out