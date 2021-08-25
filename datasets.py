import os
import json
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

def collate_fn(batch):
    assert batch

    img = torch.stack([b[0] for b in batch])
    extra = dict(
        mask = torch.stack([b[1]['mask'] for b in batch]), 
        category = [b[1]['category'] for b in batch], 
        shape = [b[1]['shape'] for b in batch], 
        image = [b[1]['image'] for b in batch], 
        shape_idx = torch.tensor([b[1]['shape_idx'] for b in batch]), 
        category_idx = torch.tensor([b[1]['category_idx'] for b in batch]), 
        bbox = torch.tensor([b[1]['bbox'] for b in batch])
    )
    views = torch.stack([b[2] for b in batch]) if len(batch[0]) > 2 else ()

    return (img, extra) + views

class RenderedViewsRandomSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_rendered_views):
        self.num_examples = num_examples
        self.num_rendered_views = num_rendered_views
        self.shuffled = None

    def set_epoch(self, epoch):
        rng = torch.Generator()
        rng.manual_seed(epoch)

        example_idx   = torch.arange(self.num_examples, dtype = torch.int64)[:, None]
        main_view_idx = torch. zeros(self.num_examples, dtype = torch.int64)[:, None]
        novel_view_idx = 1 + torch.rand(self.num_examples, self.num_rendered_views, generator = rng).argsort(-1)

        self.shuffled = torch.cat([example_idx, main_view_idx, novel_view_idx], dim = -1)
        
    def __iter__(self):
        return iter(self.shuffled.tolist())

    def __len__(self):
        return len(self.shuffled)

class RenderedViews(torchvision.datasets.VisionDataset):
    def __init__(self, root, dataset, ext = '.png'):
        super().__init__(root = root)
        self.dataset = dataset
        self.ext = ext

    def __getitem__(self, idx):
        img, extra = self.dataset[idx[0]]
        view_dir = os.path.join(self.root, extra['shape'])
        views = torch.stack([torchvision.io.read_image(os.path.join(self.root, extra['image']) if k == 0 else os.path.join(view_dir, str(k) + self.ext)) for k in idx[1:]])

        return img, extra, views

    def __len__(self):
        return len(self.dataset)

class Pix3D(torchvision.datasets.VisionDataset):
    categories           = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    categories_coco_inds = [65   , -1        , 63      , -1   , -1    ,  63   , 67     , -1    ,  -1       ]

    def __init__(self, root, metadata_path = None, max_image_size = (640, 480), target_image_size = (320, 240), **kwargs):
        super().__init__(root = root, **kwargs)
        metadata_full = json.load(open(os.path.join(root, 'pix3d.json')))
        self.shapes = {t : i for i, t in enumerate(sorted(set(m['model'] for m in metadata_full)))}
        
        self.metadata = json.load(open(metadata_path)) if metadata_path else metadata_full
        self.metadata = [m for m in self.metadata if m['img_size'][0] <= max_image_size[0] and m['img_size'][1] <= max_image_size[1]] if max_image_size and sum(max_image_size) else self.metadata

        categories = [m['category'] for m in self.metadata]
        self.num_metadata = {category : categories.count(category) for category in self.categories}
        
        self.width_min_max  = (min(m['img_size'][0] for m in self.metadata), max(m['img_size'][0] for m in self.metadata))
        self.height_min_max = (min(m['img_size'][1] for m in self.metadata), max(m['img_size'][1] for m in self.metadata))
        self.target_image_size = target_image_size

    def __getitem__(self, idx):
        m = self.metadata[idx]
        img = torchvision.io.read_image(os.path.join(self.root, m['img']))
        mask = torchvision.io.read_image(os.path.join(self.root, m['mask']))
        bbox = m['bbox']

        if self.target_image_size and sum(self.target_image_size):
            scale_factor = min(self.target_image_size[0] / img.shape[-1], self.target_image_size[1] / img.shape[-2])
            img = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0)
            mask = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0)
            #img = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
            #mask = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
            bbox = [bbox[0] * scale_factor, bbox[1] * scale_factor, bbox[2] * scale_factor, bbox[3] * scale_factor]
        
        extra = dict(
            mask = mask, 
            category = m['category'], 
            image = m['img'],
            shape = m['model'], 
            shape_idx = self.shapes[m['model']],
            category_idx = self.categories.index(m['category']), 
            bbox = bbox
        )

        return img, extra

    def __len__(self):
        return len(self.metadata)
