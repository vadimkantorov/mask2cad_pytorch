import os
import json
import collections
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
    

class RenderedViews(torchvision.datasets.VisionDataset):
    def __init__(self, root, clustered_rotations_path, dataset, transforms = None, ext = '.jpg'):
        super().__init__(root = root, transforms = transforms)
        self.dataset = dataset
        self.ext = ext
        self.clustered_rotations = torch.tensor(list(map(json.load(open(clustered_rotations_path)).get, dataset.categories)), dtype = torch.float32)

    def __getitem__(self, idx):
        images, targets = self.dataset[idx[0]]
        view_dir = os.path.join(self.root, targets['shape_path'])

        or_jpg = lambda path, ext = '.png': torchvision.io.read_image(path if os.path.exists(path) else path.replace(ext, '.jpg'))
        no_img = lambda idx: [k for k in idx if k > 0]
        # TODO: rerender to eliminate fixup
        fixup = lambda path: path if os.path.exists(path) else os.path.join(os.path.dirname(os.path.dirname(path)), 'model.obj', os.path.basename(path))
        
        views = torch.stack([or_jpg(os.path.join(self.root, targets['image_id']) if k == 0 else fixup(os.path.join(view_dir, f'{k:04}' + self.ext))) for k in no_img(idx[1:])])

        targets['shape_views'] = views.expand(-1, 3, -1, -1) / 255.0

        return img, targets 

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

class UniqueShapeRenderedViewsSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_rendered_views):
        shape2idx = {m['model'] : i for i, m in enumerate(dataset.dataset.metadata)}
        self.idx = torch.cat([torch.tensor(list(shape2idx.values()), dtype = torch.int64).unsqueeze(-1), torch.arange(1, 1 + num_rendered_views, dtype = torch.int64).repeat(len(shape2idx), 1)], dim = -1)
        
    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)

    def __len__(self):
        return len(self.dataset)
    


def collate_fn(batch):
    assert batch

    images = torch.stack([b[0] for b in batch])
    
    targets = dict(
        image_id = [b[1]['image_id'] for b in batch], 
        shape_path = [b[1]['shape_path'] for b in batch], 
        mask_path = [b[1]['mask_path'] for b in batch], 
        category = [b[1]['category'] for b in batch],

        boxes = torch.stack([b[1]['boxes'] for b in batch]),
        masks = torch.stack([b[1]['masks'] for b in batch]), 
        shape_idx = torch.stack([b[1]['shape_idx'] for b in batch]), 
        labels = torch.stack([b[1]['labels'] for b in batch]), 
        object_location = torch.stack([b[1]['object_location'] for b in batch]),
        object_rotation = torch.stack([b[1]['object_rotation'] for b in batch]),
        views = torch.stack([b[2] for b in batch]) if len(batch[0]) > 2 else None
    )
   
    return images, targets
