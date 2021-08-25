import torch
import torch.utils.data
import argparse
import datasets
import models

def main(args):
    model = models.Mask2CAD()
    model.eval()
    
    train_dataset = datasets.Pix3D(args.dataset_root, max_image_size = None)
    train_dataset = datasets.RenderedViews(args.dataset_rendered_views_root, train_dataset)

    train_sampler = datasets.RenderedViewsRandomSampler(len(train_dataset), num_rendered_views = args.num_rendered_views)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler, collate_fn = datasets.collate_fn, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
    
    breakpoint()
    train_sampler.set_epoch(0)
    for batch_idx, batch in enumerate(train_data_loader):
        print(batch_idx, batch)
        break

    #batch_img = img.unsqueeze(0) / 255.0
    #bbox = torch.tensor([[1, 1, 100, 100]], dtype = torch.float32)
    #res = model(batch_img, expand_dim(batch_img, 4, dim = 1), bbox, None, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default = 'data/common/pix3d')
    parser.add_argument('--dataset-rendered-views-root', default = 'data/pix3d_renders')
    parser.add_argument('--num-rendered-views', type = int, default = 16)
    parser.add_argument('--num-workers', type = int, default = 0)
    parser.add_argument('--train-batch-size', type = int, default = 16)
    args = parser.parse_args()

    main(args)
