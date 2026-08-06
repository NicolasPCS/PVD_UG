import torch

def as_bnc(points):
    if points.ndim != 3:
        raise ValueError("Expected a rank-3 point tensor, got {}".format(tuple(points.shape)))
    if points.shape[-1] == 3:
        return points, False
    if points.shape[1] == 3:
        return points.transpose(1,2), True
    raise ValueError("Expected point layout [B,N,3] or [B,3,N], got {}".format(tuple(points.shape)))

def pairwise_dist(x, y):
    dists = torch.cdist(x.float(), y.float())#.square()
    return dists.amin(dim=2).mean(dim=1)

def chamfer_distance(x, y):
    x_bnc, _ = as_bnc(x)
    y_bnc, _ = as_bnc(y)
    first_term = pairwise_dist(x_bnc, y_bnc)
    second_term = pairwise_dist(y_bnc, x_bnc)
    per_sample = first_term + second_term

    return per_sample