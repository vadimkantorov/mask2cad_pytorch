import os
import json
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision


def worker_init_fn(worker_id, num_threads = 1):
    torch.manual_seed(worker_id)
    #torch.cuda.manual_seed_all(worker_id) if torch.cuda.is_available() else None

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
        bbox = torch.tensor([b[1]['bbox'] for b in batch]),
        object_location = torch.tensor([b[1]['object_location'] for b in batch]),
        object_rotation = torch.tensor([b[1]['object_rotation'] for b in batch])
    )
    views = (torch.stack([b[2] for b in batch]), ) if len(batch[0]) > 2 else ()

    return (img, extra) + views

class RenderedViewsSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_rendered_views):
        self.num_examples = num_examples
        self.num_rendered_views = num_rendered_views
        self.idx = torch.arange(1, 1 + num_rendered_views).unsqueeze(0).expand(num_examples, -1)

    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

class RenderedViewsRandomSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_rendered_views, num_sampled_views, num_sampled_boxes):
        self.num_examples = num_examples
        self.num_rendered_views = num_rendered_views
        self.num_sampled_views = num_sampled_views
        self.num_sampled_boxes = num_sampled_boxes
        self.idx = None

    def set_epoch(self, epoch):
        rng = torch.Generator()
        rng.manual_seed(epoch)

        example_idx   = torch.arange(self.num_examples, dtype = torch.int64)[:, None]
        main_view_idx = torch. zeros(self.num_examples, dtype = torch.int64)[:, None]
        novel_view_idx = 1 + torch.rand(self.num_examples * self.num_sampled_boxes, self.num_rendered_views, generator = rng).argsort(-1)[..., :self.num_sampled_views].reshape(self.num_examples, -1)

        #self.idx = torch.cat([example_idx, main_view_idx, novel_view_idx], dim = -1)
        self.idx = torch.cat([example_idx, novel_view_idx], dim = -1)
        
    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

class RenderedViews(torchvision.datasets.VisionDataset):
    def __init__(self, root, clustered_rotations_path, dataset, ext = '.jpg'):
        super().__init__(root = root)
        self.dataset = dataset
        self.ext = ext
        self.clustered_rotations = torch.tensor(list(map(json.load(open(clustered_rotations_path)).get, dataset.categories)), dtype = torch.float32)

    def __getitem__(self, idx):
        img, extra = self.dataset[idx[0]]
        view_dir = os.path.join(self.root, extra['shape'])
        or_jpg = lambda path, ext = '.png': torchvision.io.read_image(path if os.path.exists(path) else path.replace(ext, '.jpg'))
        views = torch.stack([or_jpg(os.path.join(self.root, extra['image']) if k == 0 else os.path.join(view_dir, f'{k:04}' + self.ext)) for k in idx[1:]])

        return img / 255.0, extra, views.expand(-1, 3, -1, -1) / 255.0

    def __len__(self):
        return len(self.dataset)

class Pix3D(torchvision.datasets.VisionDataset):
    categories           = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    categories_coco_inds = [65   , -1        , 63      , -1   , -1    ,  63   , 67     , -1    ,  -1       ]

    def __init__(self, root, metadata_path = None, max_image_size = (640, 480), target_image_size = (320, 240), read_image = True, **kwargs):
        super().__init__(root = root, **kwargs)
        self.target_image_size = target_image_size
        self.read_image = read_image
        metadata_full = json.load(open(os.path.join(root, 'pix3d.json')))
        self.shapes = {t : i for i, t in enumerate(sorted(set(m['model'] for m in metadata_full)))}
        
        self.metadata = json.load(open(metadata_path)) if metadata_path else metadata_full
        self.metadata = [m for m in self.metadata if m['img_size'][0] <= max_image_size[0] and m['img_size'][1] <= max_image_size[1]] if max_image_size and sum(max_image_size) else self.metadata

        categories = [m['category'] for m in self.metadata]
        self.num_metadata = {category : categories.count(category) for category in self.categories}
        
        self.width_min_max  = (min(m['img_size'][0] for m in self.metadata), max(m['img_size'][0] for m in self.metadata))
        self.height_min_max = (min(m['img_size'][1] for m in self.metadata), max(m['img_size'][1] for m in self.metadata))

    def __getitem__(self, idx):
        m = self.metadata[idx]
        img_size = m['img_size']
        bbox = m['bbox']
        
        img = torchvision.io.read_image(os.path.join(self.root, m['img'])) if self.read_image else torch.empty((0, img_size[1], img_size[0]), dtype = torch.uint8)
        mask = torchvision.io.read_image(os.path.join(self.root, m['mask'])) if self.read_image else torch.empty((0, img_size[1], img_size[0]), dtype = torch.bool)

        if self.target_image_size and sum(self.target_image_size):
            scale_factor = min(self.target_image_size[0] / img.shape[-1], self.target_image_size[1] / img.shape[-2])
            img = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0) if img.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.uint8)
            mask = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0) if img.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.bool)
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
            object_location = m['trans_mat'],
            object_rotation = m['rot_mat'],
            img_width_height = img_size,
            bbox = bbox
        )

        return img, extra

    def __len__(self):
        return len(self.metadata)
