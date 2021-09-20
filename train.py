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
import json
import argparse
import datetime

import torch
import torch.utils.data
import torchvision

import pytorch3d.io

import datasets
import models
import transforms
import utils
import quat

import pix3d
import pix3d_eval
import coco_eval 

def split_list(l, n):
    cumsum = torch.tensor(n).cumsum(dim = -1).tolist()
    return [l[(cumsum[i - 1] if i >= 1 else 0) : cumsum[i]] for i in range(len(cumsum))]

def to_device(images, targets, device):
    images = images.to(device) 
    targets = {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}
    return images, targets

def from_device(outputs):
    return [{k: v.cpu() if torch.is_tensor(v) else v for k, v in t.items()} for t in outputs]

def mix_losses(loss_dict, loss_weights):
    return sum(loss_dict[k] * loss_weights[k] for k in loss_dict)
       
def recall(pred_idx, true_idx, K = 1):
    true_idx = true_idx.unsqueeze(-1) if true_idx.ndim < pred_idx.ndim else true_idx 
    assert pred_idx.shape == true_idx.shape
    return (pred_idx == true_idx).any(dim = -1).float().mean()

def train_one_epoch(log, epoch, iteration, model, optimizer, data_loader, device, epoch, print_freq, args):
    metric_logger = utils.MetricLogger()
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 1. / 1000, total_iters = min(1000, len(data_loader) - 1)) if epoch == 0 else None

    model.train()
    for images, targets in metric_logger.log_every(data_loader, print_freq, header = 'Epoch: [{}]'.format(epoch)):
        images, targets = to_device(images, targets, device = args.device)
        loss_dict = model(images, targets, mode = args.mode)
        losses = mix_losses(loss_dict, args.loss_weights) if args.mode == 'Mask2CAD' else sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss = losses_reduced, lr = optimizer.param_groups[0]['lr'], **loss_dict_reduced)
        if utils.is_main_process():
            log.write(json.dumps(dict(epoch = epoch, iteration = iteration, **metric_logger.last)) + '\n')
        iteration += 1

    return iteration

@torch.no_grad()
def evaluate(log, epoch, iteration, model, data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device, K = 10):
    metric_logger = utils.MetricLogger()
    
    #shape_retrieval = models.ShapeRetrieval(shape_data_loader, model.rendered_view_encoder)

    shape_embedding, shape_idx, shape_path = zip(*[(model.rendered_view_encoder(targets['shape_views'].flatten(end_dim = -4)), targets['shape_idx'].repeat(1, targets['shape_views'].shape[-4]).flatten(), [pp for p in targets['shape_path'] for pp in [p] * targets['shape_views'].shape[-4] ]  ) for img, targets in shape_data_loader])
    shape_embedding, shape_idx, shape_path = F.normalize(torch.cat(shape_embedding), dim = -1), torch.cat(shape_idx), [s for b in shape_path for s in b]
    shape_embedding, shape_idx, shape_path = torch.cat(utils.all_gather(shape_embedding)), torch.cat(utils.all_gather(shape_idx)), [s for ls in utils.all_gather(shape_path) for s in ls] 
    
    pred_shape_idx, true_shape_idx = utils.CatTensors(), utils.CatTensors()
    pred_category_idx, true_category_idx = utils.CatTensors(), utils.CatTensors()
    
    model.eval()
    for images, targets in metric_logger.log_every(data_loader, 100, header = 'Test:'):
        images, targets = to_device(images, targets, device = args.device)

        tic = time.time()
        detections = model(images, targets, mode = args.device)
        num_boxes = [len(d['boxes']) for d in detections]

        idx = F.normalize(torch.cat(d['shape_embedding'] for d in detections), dim = -1).matmul(shape_embedding.t()).topk(K, dim = -1, largest = True).indices
        shape_idx_, shape_path_ = self.shape_idx[idx], [self.shape_path[i[0]] for i in idx.tolist()]
        for d, s in zip(detections, split_list(shape_path_, num_boxes)):
            d['shape_path'] = s

        #outputs = from_device(outputs)
        toc_model = time.time() - tic

        pred_shape_idx.extend(split_list(shape_idx_, num_boxes))
        true_shape_idx.extend(targets['shape_idx'])
        pred_category_idx.extend(o['labels'] for o in outputs)
        true_category_idx.extend(targets['labels'])

        tic = time.time()
        evaluator_detection.update({ output['image_id']: dict(output, masks = output['masks']) for output in outputs })
        toc_evaluator_detection = time.time() - tic
        
        tic = time.time()
        evaluator_mesh.update({ output['image_id'] : dict(instances = dict(scores = output['scores'], pred_boxes = output['boxes'], pred_classes = output['labels'], pred_masks = output['masks'], pred_meshes = output['shape_path'])) for output in outputs })
        toc_evaluator_mesh = time.time() - tic

        metric_logger.update(time_model = toc_model, time_evaluator_detection = toc_evaluator_detection, time_evaluator_mesh = toc_evaluator_mesh)
    
    if args.distributed:
        pred_shape_idx.synchronize_between_processes()
        true_shape_idx.synchronize_between_processes()
        pred_category_idx.synchronize_between_processes()
        true_category_idx.synchronize_between_processes()
        metric_logger.synchronize_between_processes()
        evaluator_detection.synchronize_between_processes()
        evaluator_mesh.synchronize_between_processes()
    
    print('Averaged stats:', metric_logger)
    if utils.is_main_process():
        recall_shape = recall(pred_shape_idx.cat(), true_shape_idx.cat(), K = 5)
        recall_category = recall(pred_category_idx.cat(), true_category_idx.cat(), K = 1)
        detection_res = evaluator_detection.evaluate()
        mesh_res = evaluator_mesh.evaluate()
        line = json.dumps(dict(epoch = epoch, iteration = iteration, recall_shape = recall_shape, recall_category = recall_category, detection_res = detection_res, mesh_res = mesh_res))
        print(line)
        log.write(line + '\n')

def main(args):
    os.makedirs(args.output_path, exist_ok = True)
    utils.init_distributed_mode(args)

    log = open(args.log, 'w') if utils.is_main_process() else None

    train_dataset = pix3d.Pix3d(args.dataset_root, split_path = args.train_metadata_path, transforms = transforms.MaskRCNNAugmentations() if args.mode == 'MaskRCNN' else None)
    aspect_ratios, num_categories = train_dataset.aspect_ratios, len(train_dataset.categories)
    train_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_object_rotation_quat, train_dataset, transforms = transforms.Mask2CADAugmentations() if args.mode == 'Mask2CAD' else None)
    object_rotation_quat = train_dataset_with_views.object_rotation_quat
    
    val_dataset = pix3d.Pix3d(args.dataset_root, split_path = args.val_metadata_path)
    val_dataset_with_views = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_object_rotation_quat, val_dataset)
    
    evaluator_detection = coco_eval.CocoEvaluator(val_dataset.as_coco_dataset(), ['bbox', 'segm'])
    evaluator_mesh = pix3d_eval.Pix3dEvaluator(val_dataset)
   
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
    
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler = train_batch_sampler, num_workers = args.num_workers, collate_fn = datasets.collate_fn)
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.val_batch_size, sampler = val_sampler, num_workers = args.num_workers, collate_fn = datasets.collate_fn)
        shape_data_loader = None
    
    elif args.mode == 'Mask2CAD':

        train_sampler_with_views = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)
        val_sampler_with_views = datasets.RenderedViewsSequentialSampler(50, args.num_rendered_views) # len(val_dataset_with_views) 
        shape_sampler_with_views = datasets.UniqueShapeRenderedViewsSequentialSampler(val_dataset_with_views, args.num_rendered_views)
        if args.distributed: 
            train_sampler_with_views = datasets.DistributedSamplerWrapper(train_sampler_with_views)
            val_sampler_with_views = datasets.DistributedSamplerWrapper(val_sampler_with_views)
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset_with_views, sampler = train_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        val_data_loader = torch.utils.data.DataLoader(val_dataset_with_views, sampler = val_sampler_with_views, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True)
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
        return evaluate(log, args.start_epoch, 0, model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device=args.device)

    tic = time.time()
    iteration = 1
    for epoch in range(args.start_epoch, args.num_epochs):
        if train_data_loader.sampler is not None:
            (getattr(train_data_loader.sampler, 'set_epoch', None) or print)(epoch)
        iteration = train_one_epoch(log, epoch, iteration, model, optimizer, train_data_loader, args.device, epoch, args.print_freq, args)
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

        evaluate(log, epoch, iteration, model, val_data_loader, shape_data_loader, evaluator_detection, evaluator_mesh, args, device = args.device)

    print('Training time', datetime.timedelta(seconds = int(time.time() - tic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--log', default='data/log.jsonl')
    
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--train-metadata-path', default = 'data/common/pix3d_splits/pix3d_s2_train.json')
    parser.add_argument('--val-metadata-path', default = 'data/common/pix3d_splits/pix3d_s2_test.json')
    
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
    parser.add_argument('--dataset-object-rotation-quat', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-sampled-boxes', type = int, default = 8)

    args = parser.parse_args()
    
    print(args)
    main(args)
