import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Mask2CAD(nn.Module):
    def __init__(self, num_categories = 9, embed_dim = 256, num_rotation_clusters = 16):
        super().__init__()
        self.rendered_view_encoder = torchvision.models.resnet18(pretrained = False)
        self.object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
        
        num_categories_with_bg = 1 + num_categories
        conv_bn_relu = lambda in_channels = embed_dim, out_channels = embed_dim, kernel_size = 3: nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size // 2), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        self.shape_embedding_branch = nn.Sequential(*([conv_bn_relu() for k in range(3)] + [conv_bn_relu(embed_dim, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3)]))
        self.pose_classification_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embed_dim, num_categories_with_bg * num_rotation_clusters)]))
        self.pose_regression_branch = nn.Sequential(*([conv_bn_relu() for k in range(4)] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim = -3), nn.Linear(embed_dim, num_categories_with_bg * 4)]))

    def forward(self, img, category_idx, shape_idx, loss_weights = dict(shape_embedding = 0.5, pose_classification = 0.25, pose_regression = 5.0)):
        detections = self.object_detector(img)

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
        pass
    
    @staticmethod
    def shape_embedding_loss(img_region_features : 'BCR1', rendered_views_pos : 'BCRP', rendered_views_neg : 'BCRN', C = 1.5, tau = 0.15):
        Dpos : 'BRP' = F.cosine_similarity(img_region_features, rendered_views_pos) / tau
        Dneg : 'BRN' = F.cosine_similarity(img_region_features, rendered_views_neg) / tau

        loss : 'BR' = torch.log(Dpos / (Dpos + C * Dneg.sum(dim = -1, keepdim = True))).sum(dim = -1)
        return loss

if __name__ == '__main__':
    model = Mask2CAD()
    img = torchvision.io.read_image('data/common/pix3d/img/bed/0004.png')
    model.eval()
    res = model(img.unsqueeze(0) / 255.0, None, None)
    breakpoint()
