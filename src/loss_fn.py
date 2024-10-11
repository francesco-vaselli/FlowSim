"""All functions related to loss computation and optimization.
from https://github.com/YangLing0818/consistency_flow_matching
"""

import torch
import torch.optim as optim
import numpy as np

class dummy_sde():
    def __init__(self, init_type='gaussian', noise_scale=1.0):
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.consistencyfm_hyperparameters = {
            "delta": 1e-3,
            "num_segments": 3,
            "boundary": 1, # NOTE If wanting zero, use 0 but not 0. or 0.0, since the former is integar.
            "alpha": 1e-4,
        }

    @property
    def T(self):
      return 1.
  
    def get_z0(self, batch, train=True):
      n,c = batch.shape 

      if self.init_type == 'gaussian':
          ### standard gaussian #+ 0.5
          cur_shape = (n, c)
          return torch.randn(cur_shape)*self.noise_scale
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 


def get_consistency_flow_matching_loss_fn(train, reduce_mean=True, eps=1e-3, sde=dummy_sde()):
  """Create a loss function for training with rectified flow.

  Args:
    sde: An `SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  hyperparameter = sde.consistencyfm_hyperparameters

  def loss_fn(model, batch, context=None):
    """Compute the loss function.

    Args:
      model: A velocity model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    z0 = sde.get_z0(batch).to(batch.device)
    
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    r = torch.clamp(t + hyperparameter["delta"], max=1.0)

    t_expand = t.view(-1, 1).repeat(1, batch.shape[1])
    r_expand = r.view(-1, 1).repeat(1, batch.shape[1])
    xt = t_expand * batch + (1.-t_expand) * z0
    xr = r_expand * batch + (1.-r_expand) * z0
    
    segments = torch.linspace(0, 1, hyperparameter["num_segments"] + 1, device=batch.device)
    seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
    segment_ends = segments[seg_indices]
    
    segment_ends_expand = segment_ends.view(-1, 1).repeat(1, batch.shape[1])
    x_at_segment_ends = segment_ends_expand * batch + (1.-segment_ends_expand) * z0
    
    def f_euler(t_expand, segment_ends_expand, xt, vt):
      return xt + (segment_ends_expand - t_expand) * vt
    def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
      if (threshold, int) and threshold == 0:
        return x_at_segment_ends
      
      less_than_threshold = t_expand < threshold
      
      res = (
        less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
        + (~less_than_threshold) * x_at_segment_ends
        )
      return res
    
    model.train(train)
    model_fn = model
    if torch.cuda.is_available():
        rng_state = torch.cuda.get_rng_state()
    vt = model_fn(xt, flow_time=t[:, None], context=context)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state) # Shared Dropout Mask
    with torch.no_grad():
      if (isinstance(hyperparameter["boundary"], int) 
          and hyperparameter["boundary"] == 0): # when hyperparameter["boundary"] == 0, vr is not needed
        vr = None
      else:
        vr = model_fn(xr, flow_time=r[:, None], context=context)
        vr = torch.nan_to_num(vr)
      
    
    ft = f_euler(t_expand, segment_ends_expand, xt, vt)
    fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, hyperparameter["boundary"], x_at_segment_ends)

    ##### loss #####
    losses_f = torch.square(ft - fr)
    losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)
    
    
    def masked_losses_v(vt, vr, threshold, segment_ends, t):
      if (threshold, int) and threshold == 0:
        return 0
    
      less_than_threshold = t_expand < threshold
      
      far_from_segment_ends = (segment_ends - t) > 1.01 * hyperparameter["delta"]
      far_from_segment_ends = far_from_segment_ends.view(-1, 1).repeat(1, batch.shape[1])
      
      losses_v = torch.square(vt - vr)
      losses_v = less_than_threshold * far_from_segment_ends * losses_v
      losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
      
      return losses_v
    
    losses_v = masked_losses_v(vt, vr, hyperparameter["boundary"], segment_ends, t)

    loss = torch.mean(
      losses_f + hyperparameter["alpha"] * losses_v
    )
    return loss

  return loss_fn

class dummy_model(torch.nn.Module):
  def __init__(self):
    super(dummy_model, self).__init__()
    self.fc = torch.nn.Linear(11, 10)
  
  def forward(self, x, flow_time=None, context=None):
    return self.fc(torch.cat([x, flow_time], dim=-1))

def test_get_consistency_flow_matching_loss_fn():
    sde = dummy_sde()
    model = dummy_model()
    loss_fn = get_consistency_flow_matching_loss_fn(True, True, 1e-3, sde)
    batch = torch.randn(128, 10)
    loss = loss_fn(model, batch)
    assert loss.shape == ()
    print(loss)
    
def test():
    test_get_consistency_flow_matching_loss_fn()
    
if __name__ == "__main__":
    test()