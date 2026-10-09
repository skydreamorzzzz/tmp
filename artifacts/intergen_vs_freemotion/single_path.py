"""Minimal DPMotion zero-B inference port; no parameters or native edits.

Reference: skydreamorzzzz/DPMotion models/nets.py::InterDenoiser.single_forward.
Only inference dispatch is added. Sampling is native DDIM50, without graph,
RAG, adaptive CFG, spatial guidance, training, or checkpoint alterations.
"""
from types import MethodType
import torch
from torch import nn
from models.cfg_sampler import ClassifierFreeSampleModel
from models.gaussian_diffusion import (MotionDiffusion, space_timesteps,
    ModelMeanType, ModelVarType, LossType)


def single_forward(self, x, timesteps, mask=None, cond=None, b=None):
    B,T=x.shape[:2]
    x_a=x
    x_b=torch.zeros_like(x_a) if b is None else b
    if mask is not None:mask=mask[...,0]
    emb=self.embed_timestep(timesteps)+self.text_embed(cond)
    h_a_prev=self.sequence_pos_encoder(self.motion_embed(x_a))
    h_b_prev=self.sequence_pos_encoder(self.motion_embed(x_b))
    if mask is None:mask=torch.ones(B,T,device=x_a.device)
    key_padding_mask=~(mask>0.5)
    for block in self.blocks:
        h_a=block(h_a_prev,h_b_prev,emb,key_padding_mask)
        h_b=block(h_b_prev,h_a_prev,emb,key_padding_mask)
        h_a_prev,h_b_prev=h_a,h_b
    return self.out(h_a)


class SingleCall(nn.Module):
    def __init__(self,net):
        super().__init__();self.net=net
    def forward(self,x,timesteps,mask=None,cond=None):
        return self.net.single_forward(x,timesteps,mask=mask,cond=cond,b=None)


def forward_single(self,batch):
    cond=batch['cond'];B=cond.shape[0];T=batch['motion_lens'][0]
    diffusion=MotionDiffusion(use_timesteps=space_timesteps(self.diffusion_steps,self.sampling_strategy),
        betas=self.betas,motion_rep=self.motion_rep,model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,loss_type=LossType.MSE,rescale_timesteps=False)
    cfg_model=ClassifierFreeSampleModel(SingleCall(self.net),self.cfg_weight)
    output=diffusion.ddim_sample_loop(cfg_model,(B,T,self.nfeats),clip_denoised=False,
        progress=True,model_kwargs={'mask':None,'cond':cond},x_start=None)
    return {'output':output}


def forward_test_single(self,batch):
    batch=self.text_process(batch)
    batch.update(self.decoder.forward_single(batch))
    return batch


def install_single(net):
    before={k:tuple(v.shape) for k,v in net.state_dict().items()}
    net.decoder.net.single_forward=MethodType(single_forward,net.decoder.net)
    net.decoder.forward_single=MethodType(forward_single,net.decoder)
    net.forward_test_single=MethodType(forward_test_single,net)
    assert before=={k:tuple(v.shape) for k,v in net.state_dict().items()}
    return net
