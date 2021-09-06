import torch
import torch.utils.data
import argparse

import quat
import datasets
import models
import metrics

import os
import pytorch3d.io

def main(args):
    train_dataset = datasets.Pix3D(args.dataset_root, args.train_metadata_path, max_image_size = None)
    val_rendered_view_dataset = datasets.Pix3D(args.dataset_root, args.train_metadata_path, read_image = False)
    
    val_dataset = datasets.Pix3D(args.dataset_root, args.val_metadata_path, read_image = False, read_mask = True, target_image_size = None)
    val_evaluator = metrics.Pix3DEvaluator(val_dataset)
    
    val_evaluator.clear()
    for img, extra in val_dataset:
        image_id = extra['image']
        pred_boxes = torch.tensor(extra['bbox'])[None]
        scores = torch.ones(1)
        pred_classes = torch.tensor([extra['category_idx']])[None]
        pred_masks = extra['mask']
        
        gt_mesh = pytorch3d.io.load_obj(os.path.join(val_dataset.root, extra['shape']), load_textures = False)
        pred_meshes = [(gt_mesh[0], gt_mesh[1].verts_idx)] 
        pred_dz = torch.tensor([0.3])[None]
        
        val_evaluator.append(image_id, scores = scores, pred_boxes = pred_boxes, pred_classes = pred_classes, pred_masks = pred_masks, pred_meshes = pred_meshes, pred_dz = pred_dz)
    
    results = val_evaluator()
    print(results)
    return
    

    train_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, train_dataset)
    val_rendered_view_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, val_rendered_view_dataset)

    train_sampler = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn)

    model = models.Mask2CAD(object_rotation_quat = train_dataset.clustered_rotations)
    model.train()
    
    evaluate(model, val_dataset = val_dataset, val_rendered_view_dataset = val_rendered_view_dataset)

    train_sampler.set_epoch(0)
    for batch_idx, (img, extra, views) in enumerate(train_data_loader):
        print(batch_idx)

        bbox = extra['bbox'].repeat(1, args.num_sampled_boxes, 1)
        category_idx = extra['category_idx'].repeat(1, args.num_sampled_boxes)
        shape_idx = extra['shape_idx'].repeat(1, args.num_sampled_boxes)
        object_location = extra['object_location'].repeat(1, args.num_sampled_boxes, 1)
        object_rotation_quat = quat.from_matrix(extra['object_rotation']).repeat(1, args.num_sampled_boxes, 1)

        res = model(img, 
            rendered = views, 
            category_idx = category_idx, shape_idx = shape_idx, bbox = bbox, object_location = object_location, object_rotation_quat = object_rotation_quat
            )
        break

def evaluate(model, *, val_dataset, val_rendered_view_dataset):
    val_rendered_view_sampler = datasets.RenderedViewsSequentialSampler(50, args.num_rendered_views) # len(val_rendered_view_dataset)
    val_rendered_view_data_loader = torch.utils.data.DataLoader(val_rendered_view_dataset, sampler = val_rendered_view_sampler, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn)
    
    val_data_loader = torch.utils.data.DataLoader(val_dataset, collate_fn = datasets.collate_fn, batch_size = args.val_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn, shuffle = False)
    
    shape_retrieval = models.ShapeRetrieval(val_rendered_view_data_loader, model.rendered_view_encoder)

    model.eval()
    for batch_idx, (img, extra) in enumerate(val_data_loader):
        print(batch_idx)
        
        bbox = extra['bbox'].unsqueeze(-2)
        category_idx = extra['category_idx'].unsqueeze(-1)
        detections = model(img, bbox = bbox, category_idx = category_idx, shape_retrieval = shape_retrieval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--train-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_train.json')
    parser.add_argument('--val-metadata-path', default = 'data/common/pix3d_splits/pix3d_s1_test.json')
    parser.add_argument('--dataset-rendered-views-root', default = 'data/pix3d_renders')
    parser.add_argument('--dataset-clustered-rotations', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--train-batch-size', type = int, default = 1)
    parser.add_argument('--val-batch-size', type = int, default = 1)
    parser.add_argument('--num-sampled-boxes', type = int, default = 8)
    parser.add_argument('--learning-rate', type = float, default = 0.08)
    parser.add_argument('--num-epochs', type = int, default = 1000)
    parser.add_argument('--decay-milestones', type = int, nargs = '*', default = [32_000, 40_000])
    parser.add_argument('--decay-gamma', type = float, default = 0.1)
    parser.add_argument('--device', default = 'cpu')
    args = parser.parse_args()

    main(args)
