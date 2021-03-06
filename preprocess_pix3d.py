import json
import argparse
import itertools

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import sklearn.cluster
import scipy.spatial.transform
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', '-i', default = 'data/common/pix3d/pix3d.json')
parser.add_argument('--output-path', '-o', default = 'pix3d_clustered_viewpoints.json')
parser.add_argument('-k', type = int, default = 16)
args = parser.parse_args()
    
quatpdist = lambda a: 2 * np.arccos(np.abs(np.clip(a @ a.T, -1.0, 1.0)))

meta = json.load(open(args.input_path))
key_category = lambda m: m['category']
by_category = {k : list(g) for k, g in itertools.groupby(sorted(meta, key = key_category), key = key_category)}

algo = sklearn.cluster.KMeans(n_clusters = args.k)

quat, trans_vec = {}, {}
for k, g in by_category.items():
    print(k)

    data = np.array([scipy.spatial.transform.Rotation.from_matrix(m['rot_mat']).as_quat() for m in g])
    initial_medoids = kmeans_plusplus_initializer(data, args.k).initialize(return_index=True)
    kmedoids_instance = kmedoids(quatpdist(data), initial_medoids, data_type = 'distance_matrix')
    kmedoids_instance.process()
    medoids = kmedoids_instance.get_medoids()
    quat[k] = [data[i].tolist() for i in medoids]

json.dump({k : quat[k] for k in quat}, open(args.output_path, 'w'), indent = 2)
print(args.output_path)
