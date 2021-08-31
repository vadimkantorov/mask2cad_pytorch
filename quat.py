import scipy.spatial.transform

from_matrix = lambda R: torch.stack([torch.as_tensor(scipy.spatial.transform.Rotation.from_matrix(rot_mat).as_quat(), dtype = torch.float32) for rot_mat in R.flatten(end_dim = -2)]).view(R.shape[:-2] + (4,))
quatprodinv = lambda A, B: torch.stack([torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(t) * scipy.spatial.transform.Rotation.from_quat(q).inv()).as_quat()) for q, t in zip(A.flatten(end_dim = -2), B.flatten(end_dim = -2))]).view_as(A)
quatprod = lambda A, B: torch.stack([torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(t) * scipy.spatial.transform.Rotation.from_quat(q)).as_quat()) for q, t in zip(A.flatten(end_dim = -2), B.flatten(end_dim = -2))]).view_as(A)

quatcdist = lambda A, B: A.matmul(B.transpose(-1, -2)).abs().clamp(max = 1.0).acos().mul(2)
