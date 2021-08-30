import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import scipy.spatial.transform

class Mask2CAD(nn.Module):
    def __init__(self, num_categories = 9, embedding_dim = 256, num_rotation_clusters = 16, shape_embedding_dim = 128, object_rotation_quat = None):
        super().__init__()
        # TODO: buffer?
        self.object_rotation_quat = object_rotation_quat
        self.num_rotation_clusters = num_rotation_clusters
        self.num_categories_with_bg = num_categories
        
        self.rendered_view_encoder = torchvision.models.resnet18(pretrained = False)
        self.rendered_view_encoder.fc = nn.Linear(self.rendered_view_encoder.fc.in_features, shape_embedding_dim)
        
        self.object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
        self.object_detector.roi_heads.mask_roi_pool = CacheOutput(self.object_detector.roi_heads.mask_roi_pool)
        
        conv_bn_relu = lambda in_channels = embedding_dim, out_channels = embedding_dim, kernel_size = 3: nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        self.shape_embedding_branch = nn.Sequential(*([conv_bn_relu() for k in range(3)] + [conv_bn_relu(embedding_dim, shape_embedding_dim), nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3)]))
        self.pose_classification_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories_with_bg * self.num_rotation_clusters)]))
        self.pose_refinement_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories_with_bg * 4)]))
        self.center_regression_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories_with_bg * 2)]))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, quat_fill = 0.95):
        self.pose_refinement_branch[-1].bias.zero_()
        self.pose_refinement_branch[-1].bias[3::4] = quat_fill # xyzw
    
    def make_targets(self, bbox : 'BQ4', object_location : 'BQ3', object_rotation_quat : 'BQ4', category_idx : 'BQ'):
        quatcdist = lambda A, B: A.matmul(B.transpose(-1, -2)).abs().clamp(max = 1.0).acos().mul(2)
        quatprodinv = lambda A, B: torch.stack([torch.as_tensor((scipy.spatial.transform.Rotation.from_quat(t) * scipy.spatial.transform.Rotation.from_quat(q).inv()).as_quat()) for q, t in zip(A.flatten(end_dim = -2), B.flatten(end_dim = -2))]).view_as(A)
        
        anchor_quat = self.object_rotation_quat[category_idx]
        object_rotation_bins = quatcdist(object_rotation_quat.unsqueeze(-2), anchor_quat).squeeze(-2).argmin(dim = -1)
        anchor_quat = anchor_quat.gather(-2, object_rotation_bins[:, :, None, None].expand(-1, -1, -1, 4)).squeeze(-2)
        
        # x * q = t
        # x = t * q ** -1
        object_rotation_delta = quatprodinv(anchor_quat, object_rotation_quat)

        width, height = (bbox[..., 2] - bbox[..., 0]), (bbox[..., 3] - bbox[..., 1])
        center_x, center_y = (bbox[..., 0] + width / 2), (bbox[..., 1] + height / 2)
        center_delta = torch.stack([(object_location[..., 0] - center_x) / width, (object_location[..., 1] - center_y) / height], dim = -1)

        return object_rotation_bins, object_rotation_delta, center_delta

    def forward(self, img : 'B3HW', rendered : 'BQV3HW', *, category_idx : 'BQ' = None, shape_idx : 'BQ' = None, bbox : 'BQ4' = None, object_location : 'BQ3' = None, object_rotation_quat : 'BQ4' = None, loss_weights = dict(shape_embedding = 0.5, pose_classification = 0.25, pose_regression = 5.0), P = 4, N = 8):
        B, Q = bbox.shape[:-1]
        V = rendered.shape[-4] // Q
        
        rendered_view_features = self.rendered_view_encoder(rendered.flatten(end_dim = -4)).unflatten(0, (B, Q, V))
        
        #detections = self.object_detector(img)
        #box_features = self.object_detector.roi_heads.mask_roi_pool.output
        #box_features_list = box_features.split([d['boxes'].shape[0] for d in detections], dim=0)
        # TODO: apply masks from detections
        images = self.object_detector.transform(img)[0]
        img_features = self.object_detector.backbone(images.tensors)
        box_features = self.object_detector.roi_heads.box_roi_pool(img_features, bbox.unbind(dim = 0), images.image_sizes)
        
        shape_embedding = self.shape_embedding_branch(box_features).unflatten(0, (B, Q))
        object_rotation_bins = self.pose_classification_branch(box_features).unflatten(0, (B, Q)).unflatten(-1, (self.num_categories_with_bg, self.num_rotation_clusters))
        object_rotation_delta = self.pose_refinement_branch(box_features).unflatten(0, (B, Q)).unflatten(-1, (self.num_categories_with_bg, 4))
        center_delta = self.center_regression_branch(box_features).unflatten(0, (B, Q)).unflatten(-1, (self.num_categories_with_bg, 2))
        object_rotation_bins, object_rotation_delta, center_delta = [t.gather(-2, category_idx[..., None, None].expand(-1, -1, -1, t.shape[-1])).squeeze(-2) for t in [object_rotation_bins, object_rotation_delta, center_delta]]

        target_object_rotation_bins, target_object_rotation_delta, target_center_delta = self.make_targets(bbox, object_location, object_rotation_quat, category_idx)
        
        shape_embedding_loss = self.shape_embedding_loss(shape_embedding, rendered_view_features, category_idx = category_idx, shape_idx = shape_idx, P = P, N = N)
        pose_classification_loss, pose_regression_loss, center_regression_loss = self.pose_estimation_loss(object_rotation_bins, object_rotation_delta, center_delta, target_object_rotation_bins, target_object_rotation_delta, target_center_delta)
        
        loss = loss_weights['shape_embedding'] * shape_embedding_loss + loss_weights['pose_classification'] * pose_classification_loss + loss_weights['pose_regression'] * pose_regression_loss + loss_weights['pose_regression'] * center_regression_loss
        
        return loss

    @staticmethod
    def pose_estimation_loss(pred_pose_bins, pred_object_rotation_delta, pred_center_delta, true_pose_bins, true_object_rotation_delta, true_center_delta, delta = 0.15, theta = math.pi / 6):
        
        pose_classification_loss = F.cross_entropy(pred_pose_bins, true_pose_bins)
        
        pose_regression_loss = F.huber_loss(pred_object_rotation_delta, true_object_rotation_delta, delta = delta)
        
        center_regression_loss = F.huber_loss(pred_center_delta, true_center_delta, delta = delta)

        return pose_classification_loss, pose_regression_loss, center_regression_loss

    @staticmethod
    def shape_embedding_loss(img_region_features : 'BQC', rendered_view_features : 'BQVC', category_idx : 'BQ', shape_idx : 'BQ', C = 1.5, tau = 0.15, P = 32, N = 128):
        img_region_features, rendered_view_features = F.normalize(img_region_features, dim = -1), F.normalize(rendered_view_features, dim = -1)
        
        D = torch.mm(img_region_features.flatten(end_dim = -2), rendered_view_features.flatten(end_dim = -2).t()) / tau
        
        same_category = category_idx.reshape(-1, 1) == category_idx.unsqueeze(-1).expand(-1, -1, rendered_view_features.shape[-2]).reshape(1, -1)
        same_shape = shape_idx.reshape(-1, 1) == shape_idx.unsqueeze(-1).expand(-1, -1, rendered_view_features.shape[-2]).reshape(1, -1)

        Dpos = torch.where(same_shape, D, torch.full_like(D, float('inf'))).topk(P, dim = -1, largest = False).values
        Dneg = torch.where(same_category, D, torch.full_like(D, float('-inf'))).topk(N, dim = -1, largest = True).values

        loss = -(Dpos / (Dpos + C * Dneg.sum(dim = -1, keepdim = True))).log().sum(dim = -1)

        return loss.mean()

class CacheOutput(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.output = None

    def forward(self, *args, **kwargs):
        self.output = self.module(*args, **kwargs)
        return self.output
