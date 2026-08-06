import torch
import numpy as np

from .chamfer_distance import as_bnc, chamfer_distance

def reflect_points(points, plane_normal=(1.0, 0.0, 0.0), plane_point=(0.0, 0.0, 0.0)):
    points_bnc, transposed = as_bnc(points)
    normal = torch.as_tensor(plane_normal, dtype=points_bnc.dtype, device=points_bnc.device)
    origin = torch.as_tensor(plane_point, dtype=points_bnc.dtype, device=points_bnc.device)
    origin = origin.reshape(1, 1, 3)

    signed_distance = ((points_bnc - origin) * normal.reshape(1, 1, 3)).sum(dim=2, keepdim=True)
    reflected = points_bnc - 2.0 * signed_distance * normal.reshape(1, 1, 3)
    return reflected.transpose(1, 2) if transposed else reflected

def compute_diagonal(points):
    points_bnc, _ = as_bnc(points)

    minimum = points_bnc.amin(dim=1)  # [B,3]
    maximum = points_bnc.amax(dim=1)  # [B,3]

    extent = maximum - minimum

    diagonal = extent.square().sum(dim=1).sqrt()  # [B]

    return diagonal.clamp_min(1e-12)

def reflection_symmetry_loss(points, plane_normal=(1.0, 0.0, 0.0), plane_point=(0.0, 0.0, 0.0)):
    points_bnc, _ = as_bnc(points)

    # Center each sample in x
    centroid_x = points_bnc[:, :, 0].mean(dim=1, keepdim=True)
    centered_points = points_bnc.clone()
    centered_points[:, :, 0] = (centered_points[:, :, 0] - centroid_x)

    reflected = reflect_points(points=centered_points, plane_normal=plane_normal, plane_point=plane_point)

    cd = chamfer_distance(centered_points, reflected)
    diag = compute_diagonal(centered_points).detach()
    
    return cd/diag#.square()