import os
import json
import torch.nn.functional as F
import torchvision
import torch.utils.data

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
        
        if self.target_image_size and sum(self.target_image_size):
            scale_factor = min(self.target_image_size[0] / img.shape[-1], self.target_image_size[1] / img.shape[-2])
            img = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
            mask = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
        
        return img, dict(mask = mask, category = m['category'], shape = m['model'], shape_idx = self.shapes[m['model']],  category_idx = self.categories.index(m['category']))

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch, default_collate = torch.utils.data.dataloader.default_collate):
        return default_collate([b[0] for b in batch]), dict(mask = default_collate([b[1]['mask'] for b in batch]), category = [b[1]['category'] for b in batch], shape = [b[1]['shape'] for b in batch], shape_idx = torch.tensor([b[1]['shape_idx'] for b in batch]), category_idx = torch.tensor([b[1]['category_idx'] for b in batch]))
        
if __name__ == '__main__':
    dataset = Pix3D('./data/common/pix3d', max_image_size = None)
    print(dataset.num_metadata)
    print(dataset.width_min_max, dataset.height_min_max)
    print(dataset[0])
    print(dataset.collate_fn([dataset[0], dataset[1]]))
