from __future__ import annotations
import torch
from torch.nn import functional as F

def policy_value_loss(policy_logits:torch.Tensor,value_logits:torch.Tensor,policy_target:torch.Tensor,legal_mask:torch.Tensor,value_class:torch.Tensor,*,policy_weight:float=1.,value_weight:float=1.) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
    if policy_logits.shape != policy_target.shape or legal_mask.shape != policy_logits.shape or value_logits.shape[0] != policy_logits.shape[0]: raise ValueError("incompatible RL batch shapes")
    if policy_logits.ndim!=2 or value_logits.shape!=(policy_logits.shape[0],3) or value_class.shape!=(policy_logits.shape[0],): raise ValueError("incompatible RL batch shapes")
    if legal_mask.dtype!=torch.bool or value_class.dtype!=torch.long: raise ValueError("invalid RL target dtypes")
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(value_logits).all() or not torch.isfinite(policy_target).all(): raise ValueError("non-finite model output or target")
    if (policy_target<0).any() or not torch.allclose(policy_target.sum(1),torch.ones(policy_logits.shape[0],device=policy_target.device),atol=1e-5,rtol=0): raise ValueError("policy target rows must be probability distributions")
    if not legal_mask.any(dim=1).all() or (policy_target[~legal_mask] != 0).any(): raise ValueError("policy target includes illegal actions")
    if (value_class<0).any() or (value_class>2).any(): raise ValueError("value classes must be loss/draw/win")
    masked=policy_logits.masked_fill(~legal_mask, float("-inf")); log_probs=F.log_softmax(masked,dim=1)
    policy_loss=-(policy_target*log_probs.masked_fill(~legal_mask,0)).sum(dim=1).mean(); value_loss=F.cross_entropy(value_logits,value_class); total=policy_weight*policy_loss+value_weight*value_loss
    if not torch.isfinite(total): raise ValueError("non-finite RL loss")
    return total,{"policy_loss":policy_loss,"value_loss":value_loss,"policy_entropy":-(policy_target*torch.log(policy_target.clamp_min(1e-12))).sum(1).mean(),"top1_agreement":(masked.argmax(1)==policy_target.argmax(1)).float().mean(),"value_accuracy":(value_logits.argmax(1)==value_class).float().mean()}
