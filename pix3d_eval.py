# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import pycocotools.mask

import torch
import torch.nn.functional as F
import pytorch3d.io, pytorch3d.structures, pytorch3d.utils, pytorch3d.ops 

class Pix3dEvaluator(dict):
    def __init__(self, dataset, cocoapi):
        self.dataset = dataset
        self.cocoapi = cocoapi
        self.npos = dataset.num_by_category
        self.image_root = self.dataset.root
        self.categories = self.dataset.categories
        self.mesh_cache_ = {}

    def synchronize_between_processes(self):
        for preds in utils.all_gather(self):
            super().update(preds)

    def mesh_cache(self, shape_path):
        #self.mesh_cache_ = { model_path : (mesh[0], mesh[1].verts_idx) for model_path, mesh in zip(self.dataset.shape_idx, pytorch3d.io.load_objs_as_meshes([os.path.join(self.dataset.root, model_path) for model_path in self.dataset.shape_idx], load_textures = False)) }
        if shape_path not in self.mesh_cache_:
            mesh = pytorch3d.io.load_obj(os.path.join(self.dataset.root, shape_path), load_textures = False)
            self.mesh_cache_[shape_path] = (mesh[0], mesh[1].verts_idx) 
        return self.mesh_cache_[shape_path]

    def evaluate(self, iou_thresh = 0.5):

        pix3d_metrics = evaluate_for_pix3d(list(self.values()), npos = self.npos, cocoapi = self.cocoapi, image_root = self.image_root, mesh_cache = self.mesh_cache, iou_thresh = iou_thresh, thing_dataset_id_to_contiguous_id = {k : k - 1 for k in range(len(self.categories))})
        
        print("Box  AP {:.5f}".format(pix3d_metrics["box_ap@{:.1f}".format(iou_thresh)]))
        print("Mask AP {:.5f}".format(pix3d_metrics["mask_ap@{:.1f}".format(iou_thresh)]))
        print("Mesh AP {:.5f}".format(pix3d_metrics["mesh_ap@{:.1f}".format(iou_thresh)]))
        return pix3d_metrics


def evaluate_for_pix3d(
    predictions,
    npos,
    thing_dataset_id_to_contiguous_id, #i (dict[int->int]): Used by all instance detection/segmentation tasks in the COCO format. A mapping from instance class ids in the dataset to contiguous ids in range [0, #class). Will be automatically set by the function load_coco_json.
    cocoapi,
    image_root,
    mesh_cache,
    filter_iou = 0.3,
    iou_thresh=0.5,
    mask_thresh=0.5,
    device=torch.device("cpu"),
):
    F1_TARGET = "F1@0.300000"

    # classes
    cat_ids = sorted(cocoapi.getCatIds())
    reverse_id_mapping = {v: k for k, v in thing_dataset_id_to_contiguous_id.items()}

    # initialize tensors to record box & mask AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    mask_apscores, mask_aplabels = {}, {}
    mesh_apscores, mesh_aplabels = {}, {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mask_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mask_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mesh_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mesh_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
    box_covered, mask_covered, mesh_covered = [], [], []

    for prediction_index, prediction in enumerate(predictions):
        print(prediction_index, '/', len(predictions))
        image_id = prediction["image_id"]
        image_width = cocoapi.loadImgs([image_id])[0]["width"]
        image_height = cocoapi.loadImgs([image_id])[0]["height"]
        image_size = [image_height, image_width]

        assert "instances" in prediction

        num_img_preds = len(prediction["instances"]["scores"])
        if num_img_preds == 0:
            continue

        # predictions
        scores, labels, masks_rles, boxes = map(prediction["instances"].get, ["scores", "pred_classes", "pred_masks_rle", "pred_boxes"])
        boxes = boxes.to(device)

        assert "pred_meshes" in prediction["instances"]
        meshes = [mesh_cache(shape_path) for shape_path in prediction["instances"]["pred_meshes"]]  # preditected meshes
        verts, faces = [m[0] for m in meshes], [m[1] for m in meshes]
        meshes = pytorch3d.structures.Meshes(verts, faces).to(device)
        ##assert "pred_dz" in prediction["instances"], "Z range of box not predicted"
        
        ##pred_dz = prediction["instances"]["pred_dz"]
        heights = boxes[:, 3] - boxes[:, 1]
        # NOTE see appendix for derivation of pred dz
        ##pred_dz = pred_dz[:, 0] * heights.cpu()
        #assert list(prediction["instances"]["image_size"]) == [image_height, image_width]

        # ground truth
        # anotations corresponding to image_id (aka coco image_id)
        gt_ann_ids = cocoapi.getAnnIds(imgIds=[image_id])
        assert len(gt_ann_ids) == 1  # note that pix3d has one annotation per image
        gt_anns = cocoapi.loadAnns(gt_ann_ids)[0]
        assert gt_anns["image_id"] == image_id

        # get original ground truth mask, box, label & mesh
        gt_mask_rle = gt_anns["segmentation"]
        gt_box = torch.as_tensor(gt_anns["bbox"]).reshape(-1, 4)  # xywh from coco
        gt_box = BoxMode_convert_BoxMode_XYWH_ABS__BoxMode_XYXY_ABS(gt_box)
        gt_label = gt_anns["category_id"]
        faux_gt_targets = torch.as_tensor(gt_box, dtype=torch.float32, device=device)

        # load gt mesh and extrinsics/intrinsics
        gt_R = torch.as_tensor(gt_anns["rot_mat"], device = device)
        gt_t = torch.as_tensor(gt_anns["trans_mat"], device = device)
        gt_K = torch.as_tensor(gt_anns["K"], device = device)
        gt_verts, gt_faces = mesh_cache(gt_anns["shape_path"])
        gt_verts, gt_faces = gt_verts.to(device), gt_faces.to(device)
        gt_verts = transform_verts(gt_verts, gt_R, gt_t)
        gt_zrange = torch.stack([gt_verts[:, 2].min(), gt_verts[:, 2].max()])
        gt_mesh = pytorch3d.structures.Meshes(verts=[gt_verts], faces=[gt_faces])

        # box iou
        boxiou = pairwise_iou(boxes, faux_gt_targets)

        # filter predictions with iou > filter_iou
        valid_pred_ids = boxiou > filter_iou

        # mask iou
        miou = pycocotools.mask.iou(masks_rles, [gt_mask_rle], [0])

        # # gt zrange (zrange stores min_z and max_z)
        zranges = torch.stack([gt_zrange] * len(meshes), dim=0)

        # predicted zrange (= pred_dz)
        # It's impossible to predict the center location in Z (=tc)
        # from the image. See appendix for more.
        tc = (gt_zrange[1] + gt_zrange[0]) / 2.0
        # Given a center location (tc) and a focal_length,
        ## pred_dz = pred_dz * box_h * tc / focal_length
        # See appendix for more.
        ##zranges = torch.stack(
        ##    [
        ##        torch.stack(
        ##            [tc - tc * pred_dz[i] / 2.0 / gt_K[0], tc + tc * pred_dz[i] / 2.0 / gt_K[0]]
        ##        )
        ##        for i in range(len(meshes))
        ##    ],
        ##    dim=0,
        ##)

        gt_Ks = gt_K.view(1, 3).expand(len(meshes), 3)
        
        #meshes = pytorch3d.structures.Meshes(verts=[transform_verts(v, gt_R, gt_t) for v in meshes.verts_list()], faces=meshes.faces_list())
        meshes = transform_meshes_to_camera_coord_system(meshes, boxes, zranges, gt_Ks, image_size)

        shape_metrics = compare_meshes(meshes, gt_mesh, reduce=False)

        # sort predictions in descending order
        scores_sorted, idx_sorted = torch.sort(scores, descending=True)
        for pred_id in range(num_img_preds):
            # remember we only evaluate the preds that have overlap more than
            # iou_filter with the ground truth prediction
            if valid_pred_ids[idx_sorted[pred_id], 0] == 0:
                continue
            # map to dataset category id
            pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]].item()]
            pred_miou = miou[idx_sorted[pred_id]].item()
            pred_biou = boxiou[idx_sorted[pred_id]].item()
            pred_score = scores[idx_sorted[pred_id]].view(1).to(device)
            # note that metrics returns f1 in % (=x100)
            pred_f1 = shape_metrics[F1_TARGET][idx_sorted[pred_id]].item() / 100.0

            # mask
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_miou > iou_thresh)
                and (image_id not in mask_covered)
            ):
                tpfp[0] = 1
                mask_covered.append(image_id)
            mask_apscores[pred_label].append(pred_score)
            mask_aplabels[pred_label].append(tpfp)

            # box
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_biou > iou_thresh)
                and (image_id not in box_covered)
            ):
                tpfp[0] = 1
                box_covered.append(image_id)
            box_apscores[pred_label].append(pred_score)
            box_aplabels[pred_label].append(tpfp)

            # mesh
            tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
            if (
                (pred_label == gt_label)
                and (pred_f1 > iou_thresh)
                and (image_id not in mesh_covered)
            ):
                tpfp[0] = 1
                mesh_covered.append(image_id)
            mesh_apscores[pred_label].append(pred_score)
            mesh_aplabels[pred_label].append(tpfp)

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    pix3d_metrics = {}
    boxap, maskap, meshap = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = cocoapi.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        cat_box_ap = VOCap.compute_ap(torch.cat(box_apscores[cat_id]), torch.cat(box_aplabels[cat_id]), npos[cat_id])
        boxap += cat_box_ap
        pix3d_metrics["box_ap@%.1f - %s" % (iou_thresh, cat_name)] = float(cat_box_ap)

        cat_mask_ap = VOCap.compute_ap(torch.cat(mask_apscores[cat_id]), torch.cat(mask_aplabels[cat_id]), npos[cat_id])
        maskap += cat_mask_ap
        pix3d_metrics["mask_ap@%.1f - %s" % (iou_thresh, cat_name)] = float(cat_mask_ap)

        cat_mesh_ap = VOCap.compute_ap(torch.cat(mesh_apscores[cat_id]), torch.cat(mesh_aplabels[cat_id]), npos[cat_id])
        meshap += cat_mesh_ap
        pix3d_metrics["mesh_ap@%.1f - %s" % (iou_thresh, cat_name)] = float(cat_mesh_ap)

    pix3d_metrics["box_ap@%.1f" % iou_thresh] =  float(boxap  / valid)
    pix3d_metrics["mask_ap@%.1f" % iou_thresh] = float(maskap / valid)
    pix3d_metrics["mesh_ap@%.1f" % iou_thresh] = float(meshap / valid)
    
    return pix3d_metrics

def transform_meshes_to_camera_coord_system(meshes, boxes, zranges, Ks, imsize):
    device = meshes.device
    new_verts, new_faces = [], []
    h, w = imsize
    im_size = torch.tensor([w, h], device=device).view(1, 2)
    assert len(meshes) == len(zranges)
    for i in range(len(meshes)):
        verts, faces = meshes.get_mesh_verts_faces(i)
        if verts.numel() == 0:
            verts, faces = pytorch3d.utils.ico_sphere(level=3, device=device).get_mesh_verts_faces(0)
        assert not torch.isnan(verts).any()
        assert not torch.isnan(faces).any()
        roi = boxes[i].view(1, 4)
        zrange = zranges[i].view(1, 2)
        K = Ks[i].view(1, 3)
        cub3D = box2D_to_cuboid3D(zrange, K, roi, im_size)
        txz, tyz = cuboid3D_to_unitbox3D(cub3D)

        # image to camera coords
        verts[:, 0] = -verts[:, 0]
        verts[:, 1] = -verts[:, 1]

        # transform to destination size
        xz = verts[:, [0, 2]]
        yz = verts[:, [1, 2]]
        pxz = txz.inverse(xz.view(1, -1, 2)).squeeze(0)
        pyz = tyz.inverse(yz.view(1, -1, 2)).squeeze(0)
        verts = torch.stack([pxz[:, 0], pyz[:, 0], pxz[:, 1]], dim=1).to(
            device, dtype=torch.float32
        )

        new_verts.append(verts)
        new_faces.append(faces)

    return pytorch3d.structures.Meshes(verts=new_verts, faces=new_faces)


class ProjectiveTransform(object):
    """
    Projective Transformation in PyTorch:
    Follows a similar design to skimage.ProjectiveTransform
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py#L494
    The implementation assumes batched representations,
    so every tensor is assumed to be of shape batch x dim1 x dim2 x etc.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            # default to an identity transform
            matrix = torch.eye(3).view(1, 3, 3)
        if matrix.ndim != 3 and matrix.shape[-1] != 3 and matrix.shape[-2] != 3:
            raise ValueError("Shape of transformation matrix should be Bx3x3")
        self.params = matrix

    @property
    def _inv_matrix(self):
        return torch.inverse(self.params)

    def _apply_mat(self, coords, matrix):
        """
        Applies matrix transformation
        Input:
            coords: FloatTensor of shape BxNx2
            matrix: FloatTensor of shape Bx3x3
        Returns:
            new_coords: FloatTensor of shape BxNx2
        """
        if coords.shape[0] != matrix.shape[0]:
            raise ValueError("Mismatch in the batch dimension")
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError("Input tensors should be of shape BxNx2")

        # append 1s, shape: BxNx2 -> BxNx3
        src = torch.cat(
            [
                coords,
                torch.ones(
                    (coords.shape[0], coords.shape[1], 1), device=coords.device, dtype=torch.float32
                ),
            ],
            dim=2,
        )
        dst = torch.bmm(matrix, src.transpose(1, 2)).transpose(1, 2)
        # rescale to homogeneous coordinates
        dst[:, :, 0] /= dst[:, :, 2]
        dst[:, :, 1] /= dst[:, :, 2]

        return dst[:, :, :2]

    def __call__(self, coords):
        """Apply forward transformation.
        Input:
            coords: FloatTensor of shape BxNx2
        Output:
            coords: FloateTensor of shape BxNx2
        """
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        """Apply inverse transformation.
        Input:
            coords: FloatTensor of shape BxNx2
        Output:
            coords: FloatTensor of shape BxNx2
        """
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst, method="svd"):
        """
        Estimates the matrix to transform src to dst.
        Input:
            src: FloatTensor of shape BxNx2
            dst: FloatTensor of shape BxNx2
            method: Specifies the method to solve the linear system
        """
        if src.shape != dst.shape:
            raise ValueError("src and dst tensors but be of same shape")
        if src.ndim != 3 or src.shape[-1] != 2:
            raise ValueError("Input should be of shape BxNx2")
        device = src.device
        batch = src.shape[0]

        # Center and normalize image points for better numerical stability.
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = torch.zeros((batch, 3, 3), device=device)
            return False

        xs = src[:, :, 0]
        ys = src[:, :, 1]
        xd = dst[:, :, 0]
        yd = dst[:, :, 1]
        rows = src.shape[1]

        # params: a0, a1, a2, b0, b1, b2, c0, c1, (c3=1)
        A = torch.zeros((batch, rows * 2, 9), device=device, dtype=torch.float32)
        A[:, :rows, 0] = xs
        A[:, :rows, 1] = ys
        A[:, :rows, 2] = 1
        A[:, :rows, 6] = -xd * xs
        A[:, :rows, 7] = -xd * ys
        A[:, rows:, 3] = xs
        A[:, rows:, 4] = ys
        A[:, rows:, 5] = 1
        A[:, rows:, 6] = -yd * xs
        A[:, rows:, 7] = -yd * ys
        A[:, :rows, 8] = xd
        A[:, rows:, 8] = yd

        if method == "svd":
            A = A.cpu()  # faster computation in cpu
            # Solve for the nullspace of the constraint matrix.
            _, _, V = torch.svd(A, some=False)
            V = V.transpose(1, 2)

            H = torch.ones((batch, 9), device=device, dtype=torch.float32)
            H[:, :-1] = -V[:, -1, :-1] / V[:, -1, -1].view(-1, 1)
            H = H.reshape(batch, 3, 3)
            # H[:, 2, 2] = 1.0
        elif method == "least_sqr":
            A = A.cpu()  # faster computation in cpu
            # Least square solution
            x, _ = torch.solve(-A[:, :, -1].view(-1, 1), A[:, :, :-1])
            H = torch.cat([-x, torch.ones((1, 1), dtype=x.dtype, device=device)])
            H = H.reshape(3, 3)
        elif method == "inv":
            # x = inv(A'A)*A'*b
            invAtA = torch.inverse(torch.mm(A[:, :-1].t(), A[:, :-1]))
            Atb = torch.mm(A[:, :-1].t(), -A[:, -1].view(-1, 1))
            x = torch.mm(invAtA, Atb)
            H = torch.cat([-x, torch.ones((1, 1), dtype=x.dtype, device=device)])
            H = H.reshape(3, 3)
        else:
            raise ValueError("method {} undefined".format(method))

        # De-center and de-normalize
        self.params = torch.bmm(torch.bmm(torch.inverse(dst_matrix), H), src_matrix)
        return True


def _center_and_normalize_points(points):
    """Center and normalize points.
    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).
    Inputs:
        points: FloatTensor of shape BxNx2 of the coordinates of the image points.
    Outputs:
        matrix: FloatTensor of shape Bx3x3 of the transformation matrix to obtain
                the new points.
        new_points: FloatTensor of shape BxNx2 of the transformed image points.
    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.
    """
    device = points.device
    centroid = torch.mean(points, 1, keepdim=True)

    rms = torch.sqrt(torch.sum((points - centroid) ** 2.0, dim=(1, 2)) / points.shape[1])

    norm_factor = torch.sqrt(torch.tensor([2.0], device=device)) / rms

    matrix = torch.zeros((points.shape[0], 3, 3), dtype=torch.float32, device=device)
    matrix[:, 0, 0] = norm_factor
    matrix[:, 0, 2] = -norm_factor * centroid[:, 0, 0]
    matrix[:, 1, 1] = norm_factor
    matrix[:, 1, 2] = -norm_factor * centroid[:, 0, 1]
    matrix[:, 2, 2] = 1.0

    # matrix = torch.tensor(
    #     [
    #        [norm_factor, 0.0, -norm_factor * centroid[0]],
    #        [0.0, norm_factor, -norm_factor * centroid[1]],
    #        [0.0, 0.0, 1.0],
    #    ], device=device, dtype=torch.float32)

    pointsh = torch.cat(
        [
            points,
            torch.ones((points.shape[0], points.shape[1], 1), device=device, dtype=torch.float32),
        ],
        dim=2,
    )

    new_pointsh = torch.bmm(matrix, pointsh.transpose(1, 2)).transpose(1, 2)

    new_points = new_pointsh[:, :, :2]
    new_points[:, :, 0] /= new_pointsh[:, :, 2]
    new_points[:, :, 1] /= new_pointsh[:, :, 2]

    return matrix, new_points

def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = verts.clone().t()
    if R is not None:
        assert R.dim() == 2
        assert R.size(0) == 3 and R.size(1) == 3
        rot_verts = torch.mm(R, rot_verts)
    if t is not None:
        assert t.dim() == 1
        assert t.size(0) == 3
        rot_verts = rot_verts + t.unsqueeze(1)
    rot_verts = rot_verts.t()
    return rot_verts

def box2D_to_cuboid3D(zranges, Ks, boxes, im_sizes):
    device = boxes.device
    if boxes.shape[0] != Ks.shape[0] != zranges.shape[0]:
        raise ValueError("Ks, boxes and zranges must have the same batch dimension")
    if zranges.shape[1] != 2:
        raise ValueError("zrange must have two entries per example")
    w, h = im_sizes.t()
    sx, px, py = Ks.t()
    sy = sx
    x1, y1, x2, y2 = boxes.t()
    # transform 2d box from image coordinates to world coordinates
    x1 = w - 1 - x1 - px
    y1 = h - 1 - y1 - py
    x2 = w - 1 - x2 - px
    y2 = h - 1 - y2 - py

    cub3D = torch.zeros((boxes.shape[0], 5, 2), device=device, dtype=torch.float32)
    for i in range(2):
        z = zranges[:, i]
        x3D_min = x2 * z / sx
        x3D_max = x1 * z / sx
        y3D_min = y2 * z / sy
        y3D_max = y1 * z / sy
        cub3D[:, i * 2 + 0, 0] = x3D_min
        cub3D[:, i * 2 + 0, 1] = x3D_max
        cub3D[:, i * 2 + 1, 0] = y3D_min
        cub3D[:, i * 2 + 1, 1] = y3D_max
    cub3D[:, 4, 0] = zranges[:, 0]
    cub3D[:, 4, 1] = zranges[:, 1]
    return cub3D

def cuboid3D_to_unitbox3D(cub3D):
    device = cub3D.device
    dst = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], device=device, dtype=torch.float32
    )
    dst = dst.view(1, 4, 2).expand(cub3D.shape[0], -1, -1)
    # for (x,z) plane
    txz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack([cub3D[:, 0, 0], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 0, 1], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 2, 0], cub3D[:, 4, 1]], dim=1),
            torch.stack([cub3D[:, 2, 1], cub3D[:, 4, 1]], dim=1),
        ],
        dim=1,
    )
    if not txz.estimate(src, dst):
        raise ValueError("Estimate failed")
    # for (y,z) plane
    tyz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack([cub3D[:, 1, 0], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 1, 1], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 3, 0], cub3D[:, 4, 1]], dim=1),
            torch.stack([cub3D[:, 3, 1], cub3D[:, 4, 1]], dim=1),
        ],
        dim=1,
    )
    if not tyz.estimate(src, dst):
        raise ValueError("Estimate failed")
    return txz, tyz

@torch.no_grad()
def compare_meshes(
    pred_meshes, gt_meshes, num_samples=10000, scale="gt-10", thresholds=None, reduce=True, eps=1e-8
):
    """
    Compute evaluation metrics to compare meshes. We currently compute the
    following metrics:
    - L2 Chamfer distance
    - Normal consistency
    - Absolute normal consistency
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    Inputs:
        - pred_meshes (Meshes): Contains N predicted meshes
        - gt_meshes (Meshes): Contains 1 or N ground-truth meshes. If gt_meshes
          contains 1 mesh, it is replicated N times.
        - num_samples: The number of samples to take on the surface of each mesh.
          This can be one of the following:
            - (int): Take that many uniform samples from the surface of the mesh
            - 'verts': Use the vertex positions as samples for each mesh
            - A tuple of length 2: To use different sampling strategies for the
              predicted and ground-truth meshes (respectively).
        - scale: How to scale the predicted and ground-truth meshes before comparing.
          This can be one of the following:
            - (float): Multiply the vertex positions of both meshes by this value
            - A tuple of two floats: Multiply the vertex positions of the predicted
              and ground-truth meshes by these two different values
            - A string of the form 'gt-[SCALE]', where [SCALE] is a float literal.
              In this case, each (predicted, ground-truth) pair is scaled differently,
              so that bounding box of the (rescaled) ground-truth mesh has longest
              edge length [SCALE].
        - thresholds: The distance thresholds to use when computing precision, recall,
          and F1 scores.
        - reduce: If True, then return the average of each metric over the batch;
          otherwise return the value of each metric between each predicted and
          ground-truth mesh.
        - eps: Small constant for numeric stability when computing F1 scores.
    Returns:
        - metrics: A dictionary mapping metric names to their values. If reduce is
          True then the values are the average value of the metric over the batch;
          otherwise the values are Tensors of shape (N,).
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    pred_meshes, gt_meshes = _scale_meshes(pred_meshes, gt_meshes, scale)

    if isinstance(num_samples, tuple):
        num_samples_pred, num_samples_gt = num_samples
    else:
        num_samples_pred = num_samples_gt = num_samples

    pred_points, pred_normals = _sample_meshes(pred_meshes, num_samples_pred)
    gt_points, gt_normals = _sample_meshes(gt_meshes, num_samples_gt)
    if pred_points is None:
        print("WARNING: Sampling predictions failed during eval")
        return None
    elif gt_points is None:
        print("WARNING: Sampling GT failed during eval")
        return None

    if len(gt_meshes) == 1:
        # (1, S, 3) to (N, S, 3)
        gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        gt_normals = gt_normals.expand(len(pred_meshes), -1, -1)

    if torch.is_tensor(pred_points) and torch.is_tensor(gt_points):
        # We can compute all metrics at once in this case
        metrics = _compute_sampling_metrics(
            pred_points, pred_normals, gt_points, gt_normals, thresholds, eps
        )
    else:
        # Slow path when taking vert samples from non-equisized meshes; we need
        # to iterate over the batch
        metrics = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics = _compute_sampling_metrics(
                cur_points_pred[None], None, cur_points_gt[None], None, thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
        metrics = {k: torch.tensor(vs) for k, vs in metrics.items()}

    if reduce:
        # Average each metric over the batch
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    return metrics


def _scale_meshes(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes


def _sample_meshes(meshes, num_samples):
    """
    Helper to either sample points uniformly from the surface of a mesh
    (with normals), or take the verts of the mesh as samples.
    Inputs:
        - meshes: A MeshList
        - num_samples: An integer, or the string 'verts'
    Outputs:
        - verts: Either a Tensor of shape (N, S, 3) if we take the same number of
          samples from each mesh; otherwise a list of length N, whose ith element
          is a Tensor of shape (S_i, 3)
        - normals: Either a Tensor of shape (N, S, 3) or None if we take verts
          as samples.
    """
    if num_samples == "verts":
        normals = None
        if meshes.equisized:
            verts = meshes.verts_batch
        else:
            verts = meshes.verts_list
    else:
        verts, normals = pytorch3d.ops.sample_points_from_meshes(meshes, num_samples, return_normals=True)
    return verts, normals


def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
    """
    Compute metrics that are based on sampling points and normals:
    - L2 Chamfer distance
    - Precision at various thresholds
    - Recall at various thresholds
    - F1 score at various thresholds
    - Normal consistency (if normals are provided)
    - Absolute normal consistency (if normals are provided)
    Inputs:
        - pred_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each predicted mesh
        - pred_normals: Tensor of shape (N, S, 3) giving normals of points sampled
          from the predicted mesh, or None if such normals are not available
        - gt_points: Tensor of shape (N, S, 3) giving coordinates of sampled points
          for each ground-truth mesh
        - gt_normals: Tensor of shape (N, S, 3) giving normals of points sampled from
          the ground-truth verts, or None of such normals are not available
        - thresholds: Distance thresholds to use for precision / recall / F1
        - eps: epsilon value to handle numerically unstable F1 computation
    Returns:
        - metrics: A dictionary where keys are metric names and values are Tensors of
          shape (N,) giving the value of the metric for the batch
    """
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = pytorch3d.ops.knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = pytorch3d.ops.knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = pytorch3d.ops.knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = pytorch3d.ops.knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

class VOCap:
    @staticmethod
    def compute_ap(scores, labels, npos, device=None):
        if device is None:
            device = scores.device

        if len(scores) == 0:
            return 0.0
        tp = labels == 1
        fp = labels == 0
        sc = scores
        assert tp.size() == sc.size()
        assert tp.size() == fp.size()
        sc, ind = torch.sort(sc, descending=True)
        tp = tp[ind].to(dtype=torch.float32)
        fp = fp[ind].to(dtype=torch.float32)
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)

        # # Compute precision/recall
        rec = tp / npos
        prec = tp / (fp + tp)
        ap = VOCap.xVOCap(rec, prec, device)

        return ap

    @staticmethod
    def xVOCap(rec, prec, device):

        z = rec.new_zeros((1))
        o = rec.new_ones((1))
        mrec = torch.cat((z, rec, o))
        mpre = torch.cat((z, prec, z))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        I = (mrec[1:] != mrec[0:-1]).nonzero()[:, 0] + 1
        ap = 0
        for i in I:
            ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = area(boxes1)  # [N]
    area2 = area(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def pairwise_intersection(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: intersection, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection

def area(box) -> torch.Tensor:
    """
    Computes the area of all the boxes.
    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area

def BoxMode_convert_BoxMode_XYWH_ABS__BoxMode_XYXY_ABS(box):
    arr = box.clone()
    arr[:, 2] += arr[:, 0]
    arr[:, 3] += arr[:, 1]
    return arr
