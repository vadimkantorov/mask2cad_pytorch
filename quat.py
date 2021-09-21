import scipy.spatial.transform

import torch

detach_cpu = lambda tensor: tensor.detach().cpu()

from_matrix = lambda R: torch.as_tensor(scipy.spatial.transform.Rotation.from_matrix(detach_cpu(R).flatten(end_dim = -3)).as_quat(), dtype = torch.float32).view(R.shape[:-2] + (4,))

#quatprodinv = lambda A, B: torch.stack([torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(t) * scipy.spatial.transform.Rotation.from_quat(q).inv()).as_quat()) for q, t in zip(A.flatten(end_dim = -2), B.flatten(end_dim = -2))]).view_as(A)

quatprodinv = lambda q, t: torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(detach_cpu(t).flatten(end_dim = -2)) * scipy.spatial.transform.Rotation.from_quat(detach_cpu(q).flatten(end_dim = -2)).inv()).as_quat(), dtype = torch.float32).view_as(q)

quatprod = lambda q, t: torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(detach_cpu(t).flatten(end_dim = -2)) * scipy.spatial.transform.Rotation.from_quat(detach_cpu(q).flatten(end_dim = -2))).as_quat(), dtype = torch.float32).view_as(q)

quatcdist = lambda A, B: A.matmul(B.transpose(-1, -2)).abs().clamp(max = 1.0).acos().mul(2)
