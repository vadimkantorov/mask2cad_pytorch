import os
import json
import collections
import itertools

import torch
import torchvision

import pycocotools.coco, pycocotools.mask

class Pix3d(torchvision.datasets.VisionDataset):
    categories           = ['BACKGROUND', 'bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    categories_coco_inds = [0,            65   , -1        , 63      , -1   , -1    ,  63   , 67     , -1    ,  -1       ]

    def __init__(self, root, split_path = None, max_image_size = None, drop_images = ('img/table/1749.jpg', 'img/table/0045.png'), transforms = None, **kwargs):
        super().__init__(root = root, transforms = transforms, **kwargs)
        metadata_full = json.load(open(os.path.join(root, 'pix3d.json')))
        if split_path:
            split = json.load(open(split_path))
            images = {i['id'] : dict(img = i['file_name'], img_size = [i['width'], i['height']]) for i in split['images']}
            self.metadata = [dict(bbox = a['bbox'][:2] + [a['bbox'][0] + a['bbox'][2] - 1, a['bbox'][1] + a['bbox'][3] - 1], mask = a['segmentation'], model = a['model'], rot_mat = a['rot_mat'], trans_mat = a['trans_mat'], category = self.categories[a['category_id']], focal_length = a['K'][0] * 32 / images[a['image_id']]['img_size'][0], **images[a['image_id']]) for a in split['annotations']]
        else:
            self.metadata = metadata_full

        assert set(collections.Counter(m['img'] for m in metadata_full).values()) == {1}
        assert all(len(set(m['category'] for m in g)) == 1 for k, g in itertools.groupby(sorted(metadata_full, key = lambda m: m['model']), key = lambda m: m['model']))
        assert all(m['bbox'][0] <= m['bbox'][2] and m['bbox'][1] <= m['bbox'][3] for m in self.metadata)

        drop_image_size = max_image_size and sum(max_image_size)
        self.metadata = [m for m in self.metadata if (m['img'] not in drop_images) and (not drop_image_size or (m['img_size'][0] <= max_image_size[0] and m['img_size'][1] <= max_image_size[1]))] 

        self.shape_path = sorted(set(m['model'] for m in metadata_full))
        self.shape_idx    = {t        : i for i, t        in enumerate(self.shape_path)}
        self.category_idx = {category : i for i, category in enumerate(self.categories)}
        self.image_idx = {m['img'] : dict(m = m, file_name = m['img'], width = m['img_size'][0], height = m['img_size'][1]) for i, m in enumerate(self.metadata)}
        self.num_by_category = collections.Counter(self.category_idx[m['category']] for m in self.metadata)
        self.aspect_ratios = torch.tensor([width / height for m in self.metadata for width, height in [m['img_size']]], dtype = torch.float32)

    def __getitem__(self, idx, read_image = True, read_mask = True):
        m = self.metadata[idx]
        width, height = m['img_size']
        bbox = m['bbox']
        
        image = (torchvision.io.read_image(os.path.join(self.root, m['img']))[:3]  / 255.0) if read_image else torch.empty((0, height, width), dtype = torch.float32)
        mask = (torchvision.io.read_image(os.path.join(self.root, m['mask']))     == 255  ) if read_mask  else torch.empty((0, height, width), dtype = torch.bool)
        
        bbox = torch.as_tensor(bbox, dtype = torch.float32).unsqueeze(0)
        area = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        iscrowd = torch.zeros(len(bbox), dtype = torch.uint8)
        labels = torch.tensor(self.category_idx[m['category']]).unsqueeze(0)
        masks = mask.unsqueeze(0)
        object_location = torch.as_tensor(m['trans_mat'], dtype = torch.float32).unsqueeze(0)
        object_rotation = torch.as_tensor(m['rot_mat'  ], dtype = torch.float32).unsqueeze(0)
        shape_idx = torch.tensor(self.shape_idx[m['model']]).unsqueeze(0)

        target = dict(
            image_id   = m['img'],
            shape_path = m['model'],
            mask_path  = m['mask'],
            category   = m['category'],
            
            boxes = bbox, # xyxy
            area = area,
            iscrowd = iscrowd,
            labels = labels,
            masks = masks, 

            image_width_height = (width, height),
            shape_idx = shape_idx,
            object_location = object_location,
            object_rotation = object_rotation
        )
        
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.metadata)

    def as_coco_dataset(self):
        # annotation IDs need to start at 1, not 0, see https://github.com/pytorch/vision/issues/1530
        coco_dataset = pycocotools.coco.COCO()
        coco_dataset.dataset = dict(
            images = [dict(id = m['img'], height = m['img_size'][1], width = m['img_size'][0]) for m in self.metadata], 
            categories = [dict(id = category_idx, name = category) for category_idx, category in enumerate(self.categories) if category_idx >= 1], 
            annotations = [dict(
                id = 1 + image_idx, 
                image_id = m['img'],
                bbox = m['bbox'][:2] + [m['bbox'][2] - m['bbox'][0] + 1, m['bbox'][3] - m['bbox'][1] + 1], 
                iscrowd = 0, 
                area = (m['bbox'][2] - m['bbox'][0]) * (m['bbox'][3] - m['bbox'][1]), 
                category_id = self.category_idx[m['category']], 
                segmentation = pycocotools.mask.encode( torchvision.io.read_image(os.path.join(self.root, m['mask']))[0].eq(255).to(torch.uint8).t().contiguous().t().numpy() ),

                rot_mat = m['rot_mat'], 
                trans_mat = m['trans_mat'],
                K = [m['focal_length'] * m['img_size'][0] / 32, m['img_size'][0] / 2, m['img_size'][1] / 2],

                shape_path = m['model']

            ) for image_idx, m in enumerate(self.metadata)]
        )
        assert all(isinstance(ann['segmentation'], dict) for ann in coco_dataset.dataset['annotations'])
        coco_dataset.createIndex()
        return coco_dataset
