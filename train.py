import torch
import torch.utils.data
import argparse
import datasets
import models

import quat

def main(args):
    train_dataset = datasets.Pix3D(args.dataset_root, max_image_size = None)
    train_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, args.dataset_clustered_rotations, train_dataset)

    train_sampler = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views, num_sampled_views = args.num_sampled_views, num_sampled_boxes = args.num_sampled_boxes)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True, worker_init_fn = datasets.worker_init_fn)
    
    model = models.Mask2CAD(object_rotation_quat = train_dataset.clustered_rotations)
    model.train()
    
    train_sampler.set_epoch(0)
    for batch_idx, batch in enumerate(train_data_loader):
        print(batch_idx)
        img, extra, views = batch

        bbox = torch.tensor([1, 1, 100, 100], dtype = torch.float32).repeat(args.train_batch_size, args.num_sampled_boxes, 1)
        category_idx = extra['category_idx'].repeat(1, args.num_sampled_boxes)
        shape_idx = extra['shape_idx'].repeat(1, args.num_sampled_boxes)
        object_rotation_quat = quat.from_matrix(extra['object_rotation']).repeat(1, args.num_sampled_boxes, 1)
        object_location = extra['object_location'].repeat(1, args.num_sampled_boxes, 1)

        res = model(img / 255.0, rendered = views.expand(-1, -1, 3, -1, -1) / 255.0, category_idx = category_idx, shape_idx = shape_idx, bbox = bbox, object_location = object_location, object_rotation_quat = object_rotation_quat)
        break

    shape_retrieval_model = models.ShapeRetrieval(model, train_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--dataset-rendered-views-root', default = 'data/pix3d_renders')
    parser.add_argument('--dataset-clustered-rotations', default = 'pix3d_clustered_viewpoints.json')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-sampled-views', type = int, default = 3)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--train-batch-size', type = int, default = 1)
    parser.add_argument('--num-sampled-boxes', type = int, default = 8)
    parser.add_argument('--learning-rate', type = float, default = 0.08)
    parser.add_argument('--num-epochs', type = int, default = 1000)
    parser.add_argument('--decay-milestones', type = int, nargs = '*', default = [32_000, 40_000])
    parser.add_argument('--decay-gamma', type = float, default = 0.1)

    args = parser.parse_args()

    main(args)
