"""Model-agnostic Universal Guidance module for 3D Diffusion Models"""
import torch

from functools import partial
from .reflection_symmetry import reflection_symmetry_loss

class UniversalGuidance:
    """Apply forward Universal Guidance and retain per-step diagnostics."""

    def __init__(self, loss_fn, scale, name, max_grad_norm=None, apply_every=1, verbose=False):
        self.loss_fn = loss_fn
        # Guidance intensity
        self.scale = float(scale)
        self.name = name
        self.max_grad_norm = (
            None if max_grad_norm is None or max_grad_norm <= 0 else float(max_grad_norm)
        )
        self.apply_every = int(apply_every)
        self.losses_per_step = []
        self.sampler_calls = 0
        self.verbose = verbose

    def guide_noise(self, x_t, timestep, alpha_bar, predict_eps, decode_x0):
        """Return the UG-corrected epsilon prediction.
        
        The loss gradient includes both the denoiser Jacobian and, when presente, the decoder Jacobian because both are evaluated indise enable_grad.
        """
        call_index = self.sampler_calls
        self.sampler_calls += 1
        if call_index % self.apply_every != 0:
            return predict_eps(x_t)

        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            # Predict the noise
            eps_prediction = predict_eps(x_in)
            if eps_prediction.shape != x_in.shape:
                raise ValueError("[UG] epsilon prediction's shape {} does not match x_t's shape {}.".format(tuple(eps_prediction.shape), tuple(x_in.shape)))

            alpha = torch.as_tensor(alpha_bar, dtype=x_in.dtype, device=x_in.device)
            sqrt_alpha = alpha.clamp_min(1e-12).sqrt()
            sqrt_one_minus_alpha = (1.0 - alpha).clamp_min(0.0).sqrt()
            # Reconstruc the clean stimated object (this is the mean)
            pred_x0 = (x_in - sqrt_one_minus_alpha * eps_prediction) / sqrt_alpha
            # Convert [B,3,N] to [B,N,3] and compute symmetry loss
            decoded_points = decode_x0(pred_x0)
            loss_per_sample = self.loss_fn(decoded_points)
            if loss_per_sample.ndim == 0:
                loss_per_sample = loss_per_sample.reshape(1)

            if hasattr(self, "losses_per_step"):
                self.losses_per_step.append({
                    "timestep": int(timestep[0].detach().cpu().item()),
                    "loss_per_sample": loss_per_sample.detach().cpu(),
                })

            if self.name.lower() == "baseline" or self.scale == 0.0:
                return eps_prediction.detach()

            # Compute how much x_t should change to decrease the CD
            # Como cambia L cuando se modifica cada elemento de x_in
            gradient = torch.autograd.grad(loss_per_sample.sum(), x_in)[0]
            flat_gradient = gradient.reshape(gradient.shape[0], -1)
            raw_norm = flat_gradient.norm(dim=1)

            # Compute norm and limit bigger gradients
            clipped_fraction = 0.0
            effective_gradient = gradient
            if self.max_grad_norm is not None:
                multipliers = (self.max_grad_norm / raw_norm.clamp_min(1e-12)).clamp(max=1.0)
                clipped_fraction = float((multipliers < 1.0).float().mean().detach().cpu().item())
                multiplier_shape = [gradient.shape[0]] + [1] * (gradient.ndim - 1)
                effective_gradient = gradient * multipliers.reshape(multiplier_shape)

            effective_norm = effective_gradient.reshape(effective_gradient.shape[0], -1).norm(dim=1)

            if self.verbose:
                record = {
                    "timestep": int(timestep.reshape(-1)[0].detach().cpu().item()),
                    "loss_mean": float(loss_per_sample.mean().detach().cpu().item()),
                    "loss_max": float(loss_per_sample.max().detach().cpu().item()),
                    "raw_grad_norm_mean": float(raw_norm.mean().detach().cpu().item()),
                    "raw_grad_norm_max": float(raw_norm.max().detach().cpu().item()),
                    "effective_grad_norm_mean": float(effective_norm.mean().detach().cpu().item()),
                    "effective_grad_norm_max": float(effective_norm.max().detach().cpu().item()),
                }
                print("=" * 20, " UG Record ", "=" * 20)
                print(record)

            # Correct epsilon with symmetric gradient
            guided_eps = (eps_prediction + self.scale * effective_gradient)
            return guided_eps.detach()

def make_guidance(
    kind="reflection", scale=None,
    plane_normal=(1.0, 0.0, 0.0),
    plane_point=(0.0, 0.0, 0.0),
    rotation_axis=(0.0, 1.0, 0.0),
    rotation_center=(0.0, 0.0, 0.0),
    rotation_order=2,
    max_grad_norm=0.0,
    apply_every=1,
    verbose="False"
):
    if kind.lower() in ("baseline", "reflection"):
        print(f"[DEBUG_guidance] guidance kind = {kind}")
        loss_fn = partial(
            reflection_symmetry_loss,
            plane_normal=plane_normal,
            plane_point=plane_point
        )
    elif kind.lower() == "rotation":
        raise ValueError("Not implemented yet!!!")

    return UniversalGuidance(
        loss_fn=loss_fn,
        scale=0.0 if kind.lower() == "baseline" else scale,
        name=kind.lower(),
        max_grad_norm=max_grad_norm,
        apply_every=apply_every,
        verbose=verbose
    )