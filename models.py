import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import quat

class ShapeRetrieval(nn.Module):
    def __init__(self, data_loader, rendered_view_encoder):
        super().__init__()
        self.shape_embedding, self.shape_idx, self.shape_path = zip(*[(rendered_view_encoder(targets['shape_views'].flatten(end_dim = -4)), targets['shape_idx'].repeat(1, targets['shape_views'].shape[-4]).flatten(), [pp for p in targets['shape_path'] for pp in [p] * targets['shape_views'].shape[-4] ]  ) for img, targets in data_loader])
        self.shape_embedding, self.shape_idx, self.shape_path = F.normalize(torch.cat(self.shape_embedding), dim = -1), torch.cat(self.shape_idx), [s for b in self.shape_path for s in b]
    
    def forward(self, shape_embedding, K = 10):
        idx = F.normalize(shape_embedding, dim = -1).matmul(self.shape_embedding.t()).topk(K, dim = -1, largest = True).indices
        return self.shape_idx[idx], [self.shape_path[i[0]] for i in idx.tolist()]

    def synchronize_between_processes(self):
        pass

class Mask2CAD(nn.Module):
    def __init__(self, *, num_categories = 10, embedding_dim = 256, num_rotation_clusters = 16, shape_embedding_dim = 128, num_detections_per_image = 8, object_rotation_quat = None, **kwargs_backbone):
        super().__init__()
        self.register_buffer('object_rotation_quat', object_rotation_quat)
        self.num_rotation_clusters = num_rotation_clusters
        self.num_categories = len(object_rotation_quat)
        self.num_detections_per_image = num_detections_per_image
        
        self.rendered_view_encoder = torchvision.models.resnet18(pretrained = False)
        self.rendered_view_encoder.fc = nn.Linear(self.rendered_view_encoder.fc.in_features, shape_embedding_dim)
        
        self.object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True, trainable_backbone_layers = 0)
        self.object_detector.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.object_detector.roi_heads.box_predictor.cls_score.in_features, num_categories)
        self.object_detector.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(self.object_detector.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_categories)
        self.object_detector.roi_heads.mask_predictor = CacheInputOutput(self.object_detector.roi_heads.mask_predictor)
        self.object_detector.roi_heads.mask_roi_pool = CacheInputOutput(self.object_detector.roi_heads.mask_roi_pool)
        self.object_detector.roi_heads.detections_per_img = num_detections_per_image
        
        conv_bn_relu = lambda in_channels = embedding_dim, out_channels = embedding_dim, kernel_size = 3: nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        self.shape_embedding_branch = nn.Sequential(*([conv_bn_relu() for k in range(3)] + [conv_bn_relu(embedding_dim, shape_embedding_dim), nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3)]))
        self.pose_classification_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories * self.num_rotation_clusters)]))
        self.pose_refinement_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories * 4)]))
        self.center_regression_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embedding_dim, self.num_categories * 2)]))

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, quat_fill = 0.95):
        self.pose_refinement_branch[-1].bias.zero_()
        self.pose_refinement_branch[-1].bias[3::4] = quat_fill # xyzw

    def forward(self, images, targets, mode = None, Pfactor = 4, Nfactor = 8): # 4, 16
        bbox, category_idx, masks, shape_idx, object_location, object_rotation_quat = map(targets.get, ['boxes', 'labels', 'masks', 'shape_idx', 'object_location', 'object_rotation_quat'])
        
        if mode == 'MaskRCNN':
            detections = [dict(labels = l, boxes = b, masks = m) for l, b, m in zip(category_idx, bbox, masks)]
            self.object_detector.eval()
            res = self.object_detector(images, detections)
            return res
        
        # input / output boxes are xyxy
        if bbox is not None and category_idx is not None:
            num_boxes = [bbox.shape[1]] * len(bbox)
            #images_nested = self.object_detector.transform(images)[0]
            images_nested = torchvision.models.detection.image_list.ImageList(self.object_detector.transform.normalize(images), targets['image_width_height_resized'].tolist())
            img_features = self.object_detector.backbone(images_nested.tensors)
            box_features = self.object_detector.roi_heads.box_roi_pool(img_features, bbox.unbind(), images_nested.image_sizes)
            class_logits, box_regression = self.object_detector.roi_heads.box_predictor(self.object_detector.roi_heads.box_head(box_features))
            scores = F.softmax(class_logits, dim = -1)
            mask_features = self.object_detector.roi_heads.mask_roi_pool(img_features, bbox.unbind(), images_nested.image_sizes)
            mask_features = self.object_detector.roi_heads.mask_head(mask_features)
            mask_logits = self.object_detector.roi_heads.mask_predictor(mask_features)
            
            box_scores = self.index_select_batched(scores, category_idx.flatten())
            mask_probs = self.index_select_batched(mask_logits, category_idx.flatten()).sigmoid()
            detections = [dict(boxes = b, labels = c, scores = s, masks = m.unsqueeze(-3)) for b, c, s, m in zip(bbox, category_idx, box_scores.split(num_boxes), mask_probs.split(num_boxes))]
            detections = self.object_detector.transform.postprocess(detections, images_nested.image_sizes, targets['image_width_height'].tolist())

        else:
            detections = self.object_detector(images)
            box_features = self.object_detector.roi_heads.box_roi_pool(*self.object_detector.roi_heads.mask_roi_pool.args)
            mask_logits = self.object_detector.roi_heads.mask_predictor.output
            category_idx = torch.cat([d['labels'] for d in detections])
            mask_probs = self.index_select_batched(mask_logits, category_idx).sigmoid()
            bbox = torch.cat([d['boxes'] for d in detections])
        
        box_features = F.interpolate(box_features, mask_probs.shape[-2:]) * mask_probs.unsqueeze(-3)
        shape_embedding = self.shape_embedding_branch(box_features)
        object_rotation_bins = self.pose_classification_branch(box_features).unflatten(-1, (self.num_categories, self.num_rotation_clusters))
        object_rotation_delta = self.pose_refinement_branch(box_features).unflatten(-1, (self.num_categories, 4))
        center_delta = self.center_regression_branch(box_features).unflatten(-1, (self.num_categories, 2))
        #object_rotation_bins, object_rotation_delta, center_delta = [self.index_select_batched(t, category_idx) for t in [object_rotation_bins, object_rotation_delta, center_delta]]

        if self.training:
            B = images.shape[0]
            Q = category_idx.shape[-1]
            rendered = targets['shape_views']
            V = rendered.shape[-4] // Q
            rendered_view_features = self.rendered_view_encoder(rendered.flatten(end_dim = -4)).unflatten(0, (B, Q, V))
        
            target_object_rotation_bins, target_object_rotation_mask, target_object_rotation_delta, target_center_delta = self.compute_rotation_location_targets(category_idx, bbox, object_location, object_rotation_quat = quat.from_matrix(targets['object_rotation']))
            
            shape_embedding_loss = self.shape_embedding_loss(shape_embedding.unflatten(0, (B, Q)), rendered_view_features, category_idx = category_idx, shape_idx = shape_idx, P = Pfactor * Q, N = Nfactor * Q)
            
            pose_classification_loss, pose_regression_loss, center_regression_loss = self.pose_estimation_loss(self.index_select_batched(object_rotation_bins.unflatten(0, (B, Q)), category_idx), self.index_select_batched(object_rotation_delta.unflatten(0, (B, Q)), category_idx), self.index_select_batched(center_delta.unflatten(0, (B, Q)), category_idx), target_object_rotation_bins, target_object_rotation_mask, target_object_rotation_delta, target_center_delta)
            
            return dict(shape_embedding = shape_embedding_loss, pose_classification = pose_classification_loss, pose_regression = pose_regression_loss, center_regression = center_regression_loss)

        else:
            category_idx = category_idx.flatten()
            anchor_quat = self.index_select_batched(self.object_rotation_quat[category_idx], self.index_select_batched(object_rotation_bins.argmax(dim = -1), category_idx))
            object_rotation = quat.quatprod(anchor_quat, self.index_select_batched(object_rotation_delta, category_idx))
            center_xy, width_height = self.xyxy_to_cxcywh(bbox).split(2, dim = -1)
            #TODO: what should be scale of center_xy?
            object_location = center_xy + self.index_select_batched(center_delta, category_idx) * width_height
            num_boxes = [len(d['boxes']) for d in detections]
            image_id = targets['image_id'] if targets else [None] * len(images)

            for d, g, l, r, s in zip(detections, image_id, object_location.split(num_boxes), object_rotation.split(num_boxes), shape_embedding.split(num_boxes)):
                d['image_id'] = g
                d['location3d_center_xy'] = l
                d['rotation3d_quat'] = r
                d['shape_embedding'] = s

            return detections
    
    def compute_rotation_location_targets(self, category_idx : 'BQ', bbox : 'BQ4', object_location : 'BQ3', object_rotation_quat : 'BQ4', theta = math.pi / 6):
        anchor_quat = self.object_rotation_quat[category_idx]
        object_rotation_diff = quat.quatcdist(object_rotation_quat.unsqueeze(-2), anchor_quat).squeeze(-2)
        object_rotation_angle, object_rotation_bins = object_rotation_diff.min(dim = -1)
        object_rotation_mask = object_rotation_angle < theta

        # x * q = t => x = t * q ** -1
        object_rotation_delta = quat.quatprodinv(self.index_select_batched(anchor_quat, object_rotation_bins), object_rotation_quat)

        center_xy, width_height = self.xyxy_to_cxcywh(bbox).split(2, dim = -1)
        center_delta = (object_location[..., :2] - center_xy) / width_height

        return object_rotation_bins, object_rotation_mask, object_rotation_delta, center_delta

    @staticmethod
    def pose_estimation_loss(pred_object_rotation_bins, pred_object_rotation_delta, pred_center_delta, true_object_rotation_bins, true_object_rotation_mask, true_object_rotation_delta, true_center_delta, delta = 0.15):
        
        pose_classification_loss = F.cross_entropy(pred_object_rotation_bins.flatten(end_dim = -2), true_object_rotation_bins.flatten())
        
        pose_regression_loss = (F.huber_loss(pred_object_rotation_delta, true_object_rotation_delta, delta = delta, reduction = 'none').mean(dim = -1) * true_object_rotation_mask).sum() / true_object_rotation_mask.sum()
        
        center_regression_loss = F.huber_loss(pred_center_delta, true_center_delta, delta = delta)

        return pose_classification_loss, pose_regression_loss, center_regression_loss

    @staticmethod
    def shape_embedding_loss(img_region_features : 'BQC', rendered_view_features : 'BQVC', category_idx : 'BQ', shape_idx : 'BQ', C = 1.5, tau = 0.15, P = 32, N = 128):
        img_region_features, rendered_view_features = F.normalize(img_region_features, dim = -1), F.normalize(rendered_view_features, dim = -1)
        D = torch.mm(img_region_features.flatten(end_dim = -2), rendered_view_features.flatten(end_dim = -2).t()) / tau
        
        same_shape = shape_idx.reshape(-1, 1) == shape_idx.unsqueeze(-1).expand(-1, -1, rendered_view_features.shape[-2]).reshape(1, -1)
        same_category = category_idx.reshape(-1, 1) == category_idx.unsqueeze(-1).expand(-1, -1, rendered_view_features.shape[-2]).reshape(1, -1)

        Dpos = torch.where(same_shape, D, torch.full_like(D, float('inf'))).topk(P, dim = -1, largest = False).values
        Dneg = torch.where(same_category, D, torch.full_like(D, float('-inf'))).topk(N, dim = -1, largest = True).values

        loss = -(Dpos / (Dpos + C * Dneg.sum(dim = -1, keepdim = True))).log().sum(dim = -1)

        return loss.mean()

    @staticmethod
    def xyxy_to_cxcywh(bbox):
        width, height = (bbox[..., 2] - bbox[..., 0]), (bbox[..., 3] - bbox[..., 1])
        center_x, center_y = (bbox[..., 0] + width / 2), (bbox[..., 1] + height / 2)
        return torch.stack([center_x, center_y, width, height], dim = -1)

    @staticmethod
    def index_select_batched(tensor, I):
        return tensor.gather(I.ndim, I[(...,) + (None,) * (tensor.ndim - I.ndim)].expand((-1,) * (I.ndim + 1) + tensor.shape[I.ndim + 1:])).squeeze(I.ndim)
        

class CacheInputOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output = None
        self.args = ()
        self.kwargs = {}

    def forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.output = self.model(*args, **kwargs)
        return self.output
