#To run in a multi-gpu environment, use the distributed launcher::
#    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env  train.py ... --world-size $NGPU
#The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
#    --lr 0.02 --batch-size 2 --world-size 8
#If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
#On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
#    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

import os
import sys
import time
import math
import json
import argparse
import datetime

import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn.functional as F

import datasets
import models
import transforms
import utils
import quat

import pix3d
import pix3d_eval
import coco_eval 

detach_cpu = lambda tensor: tensor.detach().cpu()
to_device = lambda images, targets, device: (images.to(device), {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()})
from_device = lambda outputs: [{k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in outputs]
mix_losses = lambda loss_dict, loss_weights: sum(loss_dict[k] * loss_weights.get(k, 1.0) for k in loss_dict)

def split_list(l, n):
    cumsum = torch.tensor(n).cumsum(dim = -1).tolist()
    return [l[(cumsum[i - 1] if i >= 1 else 0) : cumsum[i]] for i in range(len(cumsum))]
       
def recall(pred_idx, true_idx, K = 1):
    true_idx = true_idx.unsqueeze(-1) if true_idx.ndim < pred_idx.ndim else true_idx 
    assert pred_idx.ndim == true_idx.ndim
    return (pred_idx == true_idx).any(dim = -1).float().mean()

def LinearLR(optimizer, start_factor, total_iters):
    # TODO: replace with torch.optim.lr_scheduler.LinearL upon a new pytorch release
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 if x >= total_iters else start_factor * (1 - x / total_iters) + x / total_iters)

def train_one_epoch(log, tensorboard, epoch, iteration, model, optimizer, train_data_loader, device, print_freq, args):
    metric_logger = utils.MetricLogger()
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    lr_scheduler_warmup = LinearLR(optimizer, start_factor = 1. / 1000, total_iters = min(1000, len(train_data_loader) - 1)) if epoch == 0 else None

    model.train()
    for images, targets in metric_logger.log_every(train_data_loader, print_freq, header = 'Epoch: [{}]'.format(epoch)):
        images, targets = to_device(images, targets, device = args.device)
        loss_dict = model(images, targets, mode = args.mode)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss = mix_losses(loss_dict, args.loss_weights) if args.mode == 'Mask2CAD' else sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()

        if utils.is_main_process():
            metric_logger.update(loss = float(mix_losses(loss_dict_reduced, args.loss_weights)), lr = optimizer.param_groups[0]['lr'], **loss_dict_reduced)
            log.write(json.dumps(dict(epoch = epoch, iteration = iteration, **metric_logger.last)) + '\n')
            #tensorboard.add_scalars('', value, iteration)
        iteration += 1

    return iteration

@torch.no_grad()
def evaluate(log, tensorboard, epoch, iteration, model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device, K = 10):
    metric_logger = utils.MetricLogger()
    
    shape_retrieval = models.ShapeRetrieval(shape_data_loader, model.rendered_view_encoder)
    shape_retrieval.synchronize_between_processes()
    
    pred_shape_idx, true_shape_idx = utils.CatTensors(), utils.CatTensors()
    pred_category_idx, true_category_idx = utils.CatTensors(), utils.CatTensors()
    
    model.eval()
    for images, targets in metric_logger.log_every(val_data_loader, 100, header = 'Test:'):

        tic = time.time()
        detections = model(*to_device(images, targets, device = args.device), mode = args.device)
        num_boxes = [len(d['boxes']) for d in detections]

        shape_idx_, shape_path_ = shape_retrieval(torch.cat([d['shape_embedding'] for d in detections]))
        for d, s in zip(detections, split_list(shape_path_, num_boxes)):
            d['shape_path'] = s
            d['segmentation'] = pix3d.mask_to_rle(d['masks'].squeeze(-3) > 0.5)

        toc_model = time.time() - tic
        detections = from_device(detections)

        pred_shape_idx.extend(split_list(shape_idx_, num_boxes))
        true_shape_idx.extend(targets['shape_idx'])
        pred_category_idx.extend(d['labels'] for d in detections)
        true_category_idx.extend(targets['labels'])
    
        tic = time.time()
        
        evaluator_detection.update({ d['image_id']: d for d in detections })
        toc_evaluator_detection = time.time() - tic
        
        tic = time.time()
        evaluator_mesh.update({ d['image_id'] : dict(image_id = d['image_id'], instances = dict(scores = d['scores'], pred_boxes = d['boxes'], pred_classes = d['labels'], pred_masks_rle = d['segmentation'], pred_meshes = d['shape_path'])) for d in detections })
        toc_evaluator_mesh = time.time() - tic

        metric_logger.update(time_model = toc_model, time_evaluator_detection = toc_evaluator_detection, time_evaluator_mesh = toc_evaluator_mesh)
        break
    
    if args.distributed:
        for obj in [pred_shape_ids, true_shape_idx, pred_category_idx, true_category_idx, metric_logger, evaluator_detection, evaluator_mesh]:
            obj.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    if utils.is_main_process():
        recall_shape = float(recall(pred_shape_idx.cat(), true_shape_idx.cat(), K = 5))
        recall_category = float(recall(pred_category_idx.cat(), true_category_idx.cat(), K = 1))
        detection_res = evaluator_detection.evaluate()
        mesh_res = evaluator_mesh.evaluate()
        line = json.dumps(dict(epoch = epoch, iteration = iteration, recall_shape = recall_shape, recall_category = recall_category, detection_res = detection_res, mesh_res = mesh_res))
        print(line)
        print(line, file = log)
        log.flush()
        tensorboard.add_scalars('evaluate', dict(recall_shape = recall_shape, recall_category = recall_category), iteration)
        tensorboard.flush()

def main(args):
    os.makedirs(args.output_path, exist_ok = True)
    utils.init_distributed_mode(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log = open(args.log, 'w') if utils.is_main_process() else None
    tensorboard = torch.utils.tensorboard.SummaryWriter(args.tensorboard) if utils.is_main_process() else None

    train_dataset = pix3d.Pix3d(args.dataset_root, split_path = args.train_metadata_path, transforms = transforms.MaskRCNNAugmentations() if args.mode == 'MaskRCNN' else None)
    train_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_object_rotation_quat, train_dataset, transforms = transforms.Mask2CADAugmentations() if args.mode == 'Mask2CAD' else None)
    aspect_ratios, object_rotation_quat = train_dataset.aspect_ratios, train_dataset_with_views.object_rotation_quat
    
    val_dataset = pix3d.Pix3d(args.dataset_root, split_path = args.val_metadata_path)
    val_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_object_rotation_quat, val_dataset, read_image = True, read_mask = True)
    shape_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_object_rotation_quat, val_dataset, read_image = False, read_mask = False)
    
    #val_dataset_as_coco = val_dataset.as_coco_dataset()
    #torch.save(val_dataset_as_coco, 'data/val_dataset_as_coco.pt')
    val_dataset_as_coco = torch.load('data/val_dataset_as_coco.pt')
    evaluator_detection = coco_eval.CocoEvaluator(val_dataset_as_coco, ['bbox', 'segm'])
    evaluator_mesh = pix3d_eval.Pix3dEvaluator(val_dataset, val_dataset_as_coco)
   
    if args.mode == 'MaskRCNN':
        
        if args.distributed: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle = False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = None

        #if args.aspect_ratio_group_factor >= 0:
        #    train_batch_sampler = datasets.GroupedBatchSampler(train_sampler, datasets.create_aspect_ratio_groups(aspect_ratios.tolist(), k = args.aspect_ratio_group_factor), args.train_batch_size)
        #else:
        #    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.train_batch_size, drop_last=True)
        #    batch_sampler = train_batch_sampler 
        train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size, num_workers = args.num_workers, collate_fn = datasets.collate_fn,  pin_memory = args.device != 'cpu')
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.val_batch_size, sampler = val_sampler, num_workers = args.num_workers, collate_fn = datasets.collate_fn, pin_memory = args.device != 'cpu')
        shape_data_loader = None
    
    elif args.mode == 'Mask2CAD':

        train_sampler_with_views = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)
        val_sampler_with_views = datasets.RenderedViewsSequentialSampler(len(val_dataset_with_views) , args.num_rendered_views)
        shape_sampler_with_views = datasets.UniqueShapeRenderedViewsSequentialSampler(val_dataset_with_views, args.num_rendered_views)
        shape_sampler_with_views.idx = shape_sampler_with_views.idx[:2]
        if args.distributed: 
            train_sampler_with_views = datasets.DistributedSamplerWrapper(train_sampler_with_views)
            val_sampler_with_views = datasets.DistributedSamplerWrapper(val_sampler_with_views)
            shape_sampler_with_views = datasets.DistributedSamplerWrapper(shape_sampler_with_views)
    
        train_data_loader = torch.utils.data.DataLoader(train_dataset_with_views, sampler = train_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = args.device != 'cpu')
        val_data_loader = torch.utils.data.DataLoader(val_dataset_with_views, sampler = val_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size,         num_workers = args.num_workers, pin_memory = args.device != 'cpu')
        shape_data_loader = torch.utils.data.DataLoader(val_dataset_with_views, sampler = shape_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.shape_batch_size,   num_workers = args.num_workers, pin_memory = args.device != 'cpu')

    model = models.Mask2CAD(num_categories = len(object_rotation_quat), object_rotation_quat = object_rotation_quat)

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
        assert args.mode == 'Mask2CAD'
        return evaluate(log, tensorboard, args.start_epoch, 0, model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device=args.device)

    tic = time.time()
    iteration = 1
    for epoch in range(args.start_epoch, args.num_epochs):
        if train_data_loader.sampler is not None:
            (getattr(train_data_loader.sampler, 'set_epoch', None) or print)(epoch)
        iteration = train_one_epoch(log, tensorboard, epoch, iteration, model, optimizer, train_data_loader, args.device, args.print_freq, args)
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

        evaluate(log, tensorboard, epoch, iteration, model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device = args.device)

    print('Training time', datetime.timedelta(seconds = int(time.time() - tic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log', default='data/log.jsonl')
    parser.add_argument('--tensorboard', default='data/tensorboard')
    parser.add_argument('--seed', type = int, default = 42)
    
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--train-metadata-path', default = 'data/common/pix3d_splits/pix3d_s2_train.json')
    parser.add_argument('--val-metadata-path', default = 'data/common/pix3d_splits/pix3d_s2_test.json')
    
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--train-batch-size', default = 2, type=int, help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--val-batch-size', type = int, default = 1)
    parser.add_argument('--shape-batch-size', type = int, default = 1)
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
    parser.add_argument('--dataset-object-rotation-quat', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-sampled-boxes', type = int, default = 1)

    args = parser.parse_args()
    
    print(args)
    main(args)
