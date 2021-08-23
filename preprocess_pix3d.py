import json
import argparse
import itertools
import sklearn.cluster
import scipy.spatial.transform
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
parser.add_argument('--output-path', '-o', default = 'pix3d_clustered_viewpoints.json')
parser.add_argument('-k', type = int, default = 16)
args = parser.parse_args()

meta = json.load(open(args.input_path))
key_category = lambda m: m['category']
by_category = {k : list(g) for k, g in itertools.groupby(sorted(meta, key = key_category), key = key_category)}

algo = sklearn.cluster.KMeans(n_clusters = args.k)

rot_mat, quat, trans_vec = {}, {}, {}
for k, g in by_category.items():
    print(k)
    data = np.array([m['rot_mat'] for m in g])
    algo.fit(data.reshape(len(data), -1))
    rot_mat[k] = algo.cluster_centers_.reshape(-1, 3, 3).tolist()
    
    data = np.array([scipy.spatial.transform.Rotation.from_matrix(m['rot_mat']).as_quat() for m in g])
    algo.fit(data.reshape(len(data), -1))
    quat[k] = algo.cluster_centers_.reshape(-1, 4).tolist()
    
    data = np.array([m['trans_mat'] for m in g])
    algo.fit(data.reshape(len(data), -1))
    trans_vec[k] = algo.cluster_centers_.reshape(-1, 3).tolist()

json.dump({k : dict(trans_vec = trans_vec[k], rot_mat = rot_mat[k], quat = quat[k]) for k in rot_mat}, open(args.output_path, 'w'), indent = 2)
print(args.output_path)
