import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CacheOutput(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.output = None

    def forward(self, *args, **kwargs):
        self.output = self.module(*args, **kwargs)
        return self.output

class Mask2CAD(nn.Module):
    def __init__(self, num_categories = 9, embed_dim = 256, num_rotation_clusters = 16):
        super().__init__()
        self.rendered_view_encoder = torchvision.models.resnet18(pretrained = False)
        self.object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
        self.object_detector.roi_heads.mask_roi_pool = CacheOutput(self.object_detector.roi_heads.mask_roi_pool)
        
        num_categories_with_bg = 1 + num_categories
        conv_bn_relu = lambda in_channels = embed_dim, out_channels = embed_dim, kernel_size = 3: nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        self.shape_embedding_branch = nn.Sequential(*([conv_bn_relu() for k in range(3)] + [conv_bn_relu(embed_dim, embed_dim // 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3)]))
        self.pose_classification_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embed_dim, num_categories_with_bg * num_rotation_clusters)]))
        self.pose_refinement_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embed_dim, num_categories_with_bg * 4)]))
        self.center_regression_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embed_dim, num_categories_with_bg * 2)]))

    def forward(self, img : 'B3HW', rendered : 'BR3HW', bbox, category_idx, shape_idx, loss_weights = dict(shape_embedding = 0.5, pose_classification = 0.25, pose_regression = 5.0)):
        rendered_features = self.rendered_view_encoder(rendered.flatten(end_dim = 1)).unflatten(0, rendered.shape[:2])
        
        #detections = self.object_detector(img)
        #box_features = self.object_detector.roi_heads.mask_roi_pool.output
        #box_features_list = box_features.split([d['boxes'].shape[0] for d in detections], dim=0)
        # TODO: apply masks from detections
        
        images = self.object_detector.transform(img)[0]
        img_features = self.object_detector.backbone(images.tensors)
        box_features = self.object_detector.roi_heads.box_roi_pool(img_features, bbox.unsqueeze(1).unbind(dim = 0), images.image_sizes)



        pose_classification = self.pose_classification_branch(box_features)
        pose_refinement = self.pose_refinement_branch(box_features)
        center_regression = self.center_regression_branch(box_features)

        shape_embedding_loss = 0
        pose_classification_loss = 0
        pose_regression_loss = 0
        
        loss = loss_weights['shape_embedding'] * shape_embedding_loss + loss_weights['pose_classification'] * pose_classification_loss + loss_weights['pose_regression'] * pose_regression_loss
        
        return detections
        
    @staticmethod
    def pose_regression_loss(pred_center_delta, pred_quat, true_center_delta, true_quat, delta = 0.15):
        # We thus regress the object center as a bounding regression problem. More specifically, for each ROI, we task the network with predicting (δx, δy), where the δs are the shift between bounding box center and actual object center as a ratio of object width and height. 
        center_regression_loss = F.huber_loss(pred_center_delta, true_center_delta, delta = delta)
        pose_regression_loss = F.huber_loss(pred_quat, true_quat, delta = delta)

        loss = center_regression_loss + pose_regression_loss
        return loss

    @staticmethod
    def pose_classification_loss(a, b):
        # cross entropy
        pass
    
    @staticmethod
    def shape_embedding_loss(img_region_features : 'BCR1', rendered_views_pos : 'BCRP', rendered_views_neg : 'BCRN', C = 1.5, tau = 0.15):
        Dpos : 'BRP' = F.cosine_similarity(img_region_features, rendered_views_pos) / tau
        Dneg : 'BRN' = F.cosine_similarity(img_region_features, rendered_views_neg) / tau

        loss : 'BR' = -torch.log(Dpos / (Dpos + C * Dneg.sum(dim = -1, keepdim = True))).sum(dim = -1)
        return loss
