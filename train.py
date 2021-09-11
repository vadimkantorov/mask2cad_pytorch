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

import os
import sys
import time
import math
import argparse
import datetime

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

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 if x >= warmup_iters else warmup_factor * (1 - x / warmup_iters) + x / warmup_iters)

def mix_losses(loss_dict, loss_weights):
    return sum(loss_dict[k] * loss_weights[k] for k in loss_dict)
       
def to_device(images, targets, device):
    images = images.to(device) 
    targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
    return images, targets

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    lr_scheduler = warmup_lr_scheduler(optimizer, min(1000, len(data_loader) - 1), warmup_factor = 1. / 1000) if epoch == 0 else None

    batch = next(iter(data_loader))
    for images, targets in metric_logger.log_every(data_loader, print_freq, header = 'Epoch: [{}]'.format(epoch)):
        images, targets = to_device(images, targets, device = args.device)

        loss_dict = model(images, targets, mode = args.mode)
        losses = mix_losses(loss_dict, args.loss_weights) if args.mode == 'Mask2CAD' else sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter = '  ')
    
    evaluator_mesh.clear()
    shape_retrieval = models.ShapeRetrieval(shape_data_loader, model.rendered_view_encoder)

    for images, targets in metric_logger.log_every(data_loader, 100, header = 'Test:'):
        images, targets = to_device(images, targets, device = args.device)

        tic = time.time()
        outputs = model(images, targets, mode = args.device, shape_retrieval = shape_retrieval)
        outputs = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in outputs]
        toc_model = time.time() - tic

        tic = time.time()
        evaluator_detection.update({output['image_id']: dict(output, masks = output['masks'][:, None]) for output in outputs})
        toc_evaluator_detection = time.time() - tic
        
        tic = time.time()
        evaluator_mesh.update({ output['image_id'] : dict(instances = dict(scores = output['scores'], pred_boxes = output['boxes'], pred_classes = output['labels'], pred_masks = output['masks'])) for output in outputs })
        # pred_meshes = pred_meshes
        toc_evaluator_mesh = time.time() - tic

        metric_logger.update(time_model = toc_model, time_evaluator_detection = toc_evaluator_detection, time_evaluator_mesh = toc_evaluator_mesh)
        break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    
    evaluator_detection.synchronize_between_processes()
    evaluator_detection.accumulate()
    
    print('Detection', evaluator_detection.summarize())
    breakpoint()
    print('Mesh', evaluator_mesh.summarize())

def build_transform(train, data_augmentation):
    return transforms.DetectionPresetTrain(data_augmentation) if train else transforms.DetectionPresetEval()
    
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    os.makedirs(args.output_path, exist_ok = True)
    utils.init_distributed_mode(args)

    train_dataset = datasets.Pix3d(args.dataset_root, split_path = args.train_metadata_path)
    aspect_ratios, num_categories = train_dataset.aspect_ratios, len(train_dataset.categories)
    train_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, train_dataset)
    object_rotation_quat = train_dataset_with_views.clustered_rotations
    
    val_dataset = datasets.Pix3d(args.dataset_root, split_path = args.val_metadata_path)
    val_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, val_dataset)
    
    evaluator_detection = coco_eval.CocoEvaluator(val_dataset.as_coco_dataset(), ['bbox', 'segm'])
    evaluator_mesh = pix3d_eval.Pix3dEvaluator(val_dataset)
   
    if args.mode == 'MaskRCNN':
        
        if args.distributed: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        if args.aspect_ratio_group_factor >= 0:
            train_batch_sampler = samplers.GroupedBatchSampler(train_sampler, samplers.create_aspect_ratio_groups(aspect_ratios.tolist(), k = args.aspect_ratio_group_factor), args.train_batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.train_batch_size, drop_last=True)
    
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = train_batch_sampler, num_workers = args.num_workers, collate_fn = datasets.collate_fn)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.val_batch_size, sampler = val_sampler, num_workers = args.num_workers, collate_fn = datasets.collate_fn)
        shape_data_loader = None
    
    elif args.mode == 'Mask2CAD':

        train_sampler_with_views = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)
        
        val_sampler_with_views = datasets.RenderedViewsSequentialSampler(50, args.num_rendered_views) # len(val_rendered_view_dataset)
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset_with_views, sampler = train_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        val_data_loader = torch.utils.data.DataLoader(val_dataset_with_views, sampler = val_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True)

        shape_sampler_with_views = datasets.UniqueShapeRenderedViewsSequentialSampler(val_dataset_with_views, args.num_rendered_views)
        shape_data_loader = torch.utils.data.DataLoader(val_dataset_with_views, sampler = shape_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True)

    model = models.Mask2CAD(num_categories = num_categories, object_rotation_quat = object_rotation_quat)

    model.to(args.device)
    if args.distributed and args.convert_sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.lr_steps, gamma = args.lr_gamma) if args.lr_scheduler == 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.num_epochs) if args.lr_scheduler == 'CosineAnnealingLR' else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.evaluate_only:
        return evaluate(model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, device=args.device)

    tic = time.time()
    for epoch in range(args.start_epoch, args.num_epochs):
        breakpoint()
        if train_data_loader.sampler is not None:
            (getattr(train_data_loader.sampler, 'set_epoch', None) or print)(epoch)
        train_one_epoch(model, optimizer, train_data_loader, args.device, epoch, args.print_freq, args)
        lr_scheduler.step()

        if False and args.output_path:
            checkpoint = dict(
                model = model_without_ddp.state_dict(),
                optimizer = optimizer.state_dict(),
                lr_scheduler = lr_scheduler.state_dict(),
                args = args,
                epoch = epoch
            )
            utils.save_on_main(checkpoint, os.path.join(args.output_path, 'model_{}.pt'.format(epoch)))
            utils.save_on_main(checkpoint, os.path.join(args.output_path, 'checkpoint.pt'))

        evaluate(model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, device = args.device)

    print('Training time', datetime.timedelta(seconds = int(time.time() - tic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--train-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_train.json')
    parser.add_argument('--val-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_test.json')
    
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
    parser.add_argument('--output-path', '-o', default = 'data')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--data-augmentation', default='hflip', help='data augmentation policy (default: hflip)')
    parser.add_argument('--convert-sync-batchnorm', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    
    parser.add_argument('--mode', default = 'MaskRCNN', choices = ['MaskRCNN', 'Mask2CAD'] )
    parser.add_argument('--loss-weights', default = dict(shape_embedding = 0.5, pose_classification = 0.25, pose_regression = 5.0, center_regression = 5.0), action = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: getattr(n, a.dest).update({k : float(v) for k, v in [v.split('=')]})))) 
    
    parser.add_argument('--dataset-rendered-views-root', default = 'data/pix3d_renders')
    parser.add_argument('--dataset-clustered-rotations', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-sampled-boxes', type = int, default = 8)
    

    args = parser.parse_args()
    
    print(args)
    main(args)
