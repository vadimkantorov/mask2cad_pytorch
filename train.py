r'''PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
'''

import argparse
import datetime
import os
import math
import sys
import time

import torch
import torch.utils.data
import torchvision

import pytorch3d.io

import datasets
import models
import samplers
import transforms
import utils
import quat

import pix3d_eval
import coco_eval 

from coco_utils import get_coco_api_from_dataset

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, sampler, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        
        object_location = extra['object_location'].repeat(1, args.num_sampled_boxes, 1)
        object_rotation_quat = quat.from_matrix(extra['object_rotation']).repeat(1, args.num_sampled_boxes, 1)
        res = model(img, 
            rendered = views, 
            category_idx = category_idx, shape_idx = shape_idx, bbox = bbox, object_location = object_location, object_rotation_quat = object_rotation_quat
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        if not torch.isfinite(losses_reduced):
            loss_value = losses_reduced.item()
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, evaluator_pix3d, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    val_rendered_view_sampler = datasets.RenderedViewsSequentialSampler(50, args.num_rendered_views) # len(val_rendered_view_dataset)
    val_rendered_view_data_loader = torch.utils.data.DataLoader(val_rendered_view_dataset, sampler = val_rendered_view_sampler, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn, shuffle = False)
    shape_retrieval = models.ShapeRetrieval(val_rendered_view_data_loader, model.rendered_view_encoder)
    
    evaluator_pix3d.clear()

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
        image_id = extra['image']
        scores = torch.ones(1)
        pred_classes = torch.tensor([extra['category_idx']])[None]
        pred_masks = extra['mask']
        
        gt_mesh = pytorch3d.io.load_obj(os.path.join(val_dataset.root, extra['shape']), load_textures = False)
        pred_meshes = [(gt_mesh[0], gt_mesh[1].verts_idx)] 
        pred_dz = torch.tensor([0.3])[None]
        
        val_evaluator.update({image_id : dict(instances = dict(scores = scores, pred_boxes = pred_boxes, pred_classes = pred_classes, pred_masks = pred_masks, pred_meshes = pred_meshes, pred_dz = pred_dz)) })
        #detections = model(img, bbox = bbox, category_idx = category_idx, shape_retrieval = shape_retrieval)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator

def build_transform(train, data_augmentation):
    return transforms.DetectionPresetTrain(data_augmentation) if train else transforms.DetectionPresetEval()
    
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def build_model(num_classes, hidden_layer = 256, trainable_backbone_layers = 0, rpn_score_thresh = float('inf')):
    model = models.Mask2CAD(object_rotation_quat = train_dataset.clustered_rotations)
    return model

def main(args):
    os.makedirs(args.output_path, exist_ok = True)
    utils.init_distributed_mode(args)

    train_dataset = datasets.Pix3D(args.dataset_root, split_path = args.train_metadata_path)
    train_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, train_dataset)
    
    val_dataset = datasets.Pix3D(args.dataset_root, split_path = args.val_metadata_path)
    
    val_rendered_view_dataset = datasets.Pix3D(args.dataset_root, args.train_metadata_path, read_image = False)
    val_rendered_view_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, val_rendered_view_dataset)
    
    num_classes = len(train_dataset.categories)
    
    train_sampler = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)
    
    evaluator_coco = coco_eval.CocoEvaluator(val_dataset.as_coco_dataset(), ['bbox', 'segm'])
    evaluator_pix3d = pix3d_eval.Pix3DEvaluator(val_dataset)
    
   if args.distributed: 
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        train_batch_sampler = samplers.GroupedBatchSampler(train_sampler, samplers.create_aspect_ratio_groups(train_dataset.aspect_ratios.tolist(), k = args.aspect_ratio_group_factor), args.train_batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.train_batch_size, drop_last=True)
    
    collate_fn = lambda batch: tuple(zip(*batch))
    train_data_loader  =  torch.utils.data.DataLoader(train_dataset, batch_sampler = train_batch_sampler, num_workers = args.num_workers, collate_fn = collate_fn)
    val_data_loader  =  torch.utils.data.DataLoader(val_dataset, batch_size = 1, sampler = val_sampler, num_workers = args.num_workers, collate_fn = collate_fn)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn)


    model = build_model(num_classes = num_classes, pretrained = args.pretrained, trainable_backbone_layers = args.trainable_backbone_layers, rpn_score_thresh = args.rpn_score_thresh if args.rpn_score_thresh is not None else None)
    model.to(args.device)
    if args.distributed and args.convert_sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_scheduler == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_steps, gamma = args.lr_gamma)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.evaluate_only:
        evaluate(model, val_data_loader, evaluator_pix3d, device=args.device)
        return

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_sampler, train_data_loader, args.device, epoch, args.print_freq)
        lr_scheduler.step()

        if args.output_path:
            checkpoint = dict(
                model = model_without_ddp.state_dict(),
                optimizer = optimizer.state_dict(),
                lr_scheduler = lr_scheduler.state_dict(),
                args = args,
                epoch = epoch
            )
            utils.save_on_main(checkpoint, os.path.join(args.output_path, 'model_{}.pt'.format(epoch)))
            utils.save_on_main(checkpoint, os.path.join(args.output_path, 'checkpoint.pt'))

        evaluate(model, val_data_loader, evaluator_pix3d, device=args.device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--train-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_train.json')
    parser.add_argument('--val-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_test.json')
    parser.add_argument('--dataset', default='Pix3D')
    
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--train-batch-size', default=2, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--val-batch-size', type = int, default = 1)
    parser.add_argument('--num-epochs', type = int, default = 1000)
    parser.add_argument('--num-workers', '-j', default=0, type=int)
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-scheduler', default = 'MultiStepLR', choices = ['MultiStepLR', 'CosineAnnealingLR'])
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--decay-milestones', type = int, nargs = '*', default = [32_000, 40_000])
    parser.add_argument('--decay-gamma', type = float, default = 0.1)

    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--output-path', '-o', default='data')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int, help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default='hflip', help='data augmentation policy (default: hflip)')
    parser.add_argument('--convert-sync-batchnorm', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--dataset-rendered-views-root', default = 'data/pix3d_renders')
    parser.add_argument('--dataset-clustered-rotations', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-sampled-boxes', type = int, default = 8)

    args = parser.parse_args()
    
    print(args)
    main(args)
