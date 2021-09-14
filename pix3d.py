import os
import json
import collections
import torchvision

import pycocotools.coco, pycocotools.mask    

class Pix3d(torchvision.datasets.VisionDataset):
    categories           = ['background', 'bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    categories_coco_inds = [0,            65   , -1        , 63      , -1   , -1    ,  63   , 67     , -1    ,  -1       ]

    def __init__(self, root, split_path = None, max_image_size = None, target_image_size = (320, 240), drop_images = ['img/table/1749.jpg', 'img/table/0045.png'], read_image = True, read_mask = True, transforms = None, **kwargs):
        super().__init__(root = root, transforms = transforms, **kwargs)
        self.target_image_size = target_image_size
        self.read_image = read_image
        self.read_mask = read_mask
        self.transforms = transforms
        metadata_full = json.load(open(os.path.join(root, 'pix3d.json')))
        
        assert set(collections.Counter(m['img'] for m in metadata_full).values()) == {1}
        
        self.shapes = sorted(set(m['model'] for m in metadata_full))
        self.shape_idx = {t        : i for i, t in enumerate(self.shapes)}
        self.category_idx = {category : i for i, category in enumerate(self.categories)}

        if split_path:
            split = json.load(open(split_path))
            images = {i['id'] : dict(img = i['file_name'], img_size = [i['width'], i['height']]) for i in split['images']}
            self.metadata = [dict(bbox = a['bbox'][:2] + [a['bbox'][0] + a['bbox'][2] - 1, a['bbox'][1] + a['bbox'][3] - 1], mask = a['segmentation'], model = a['model'], rot_mat = a['rot_mat'], trans_mat = a['trans_mat'], category = self.categories[a['category_id'] - 1], focal_length = a['K'][0] * 32 / images[a['image_id']]['img_size'][0], **images[a['image_id']]) for a in split['annotations']]
        else:
            self.metadata = metadata_full

        assert all(m['bbox'][0] <= m['bbox'][2] and m['bbox'][1] <= m['bbox'][3] for m in self.metadata)

        drop_image_size = max_image_size and sum(max_image_size)
        self.metadata = [m for m in self.metadata if (m['img'] not in drop_images) and (not drop_image_size or (m['img_size'][0] <= max_image_size[0] and m['img_size'][1] <= max_image_size[1]))] 

        self.image_idx = {m['img'] : dict(m = m, file_name = m['img'], width = m['img_size'][0], height = m['img_size'][1]) for i, m in enumerate(self.metadata)}
        self.num_by_category = collections.Counter(self.category_idx[m['category']] for m in self.metadata)
        self.width_min_max  = (min(m['img_size'][0] for m in self.metadata), max(m['img_size'][0] for m in self.metadata))
        self.height_min_max = (min(m['img_size'][1] for m in self.metadata), max(m['img_size'][1] for m in self.metadata))
        self.aspect_ratios = torch.tensor([m['img_size'][0] / m['img_size'][1] for m in self.metadata], dtype = torch.float32)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        img_size = m['img_size']
        bbox = m['bbox']
        
        img = torchvision.io.read_image(os.path.join(self.root, m['img'])) if self.read_image else torch.empty((0, img_size[1], img_size[0]), dtype = torch.uint8)
        mask = torchvision.io.read_image(os.path.join(self.root, m['mask'])) if self.read_mask else torch.empty((0, img_size[1], img_size[0]), dtype = torch.uint8)

        if self.target_image_size and sum(self.target_image_size):
            scale_factor = min(self.target_image_size[0] / img.shape[-1], self.target_image_size[1] / img.shape[-2])
            img = F.interpolate(img.unsqueeze(0), self.target_image_size).squeeze(0) if img.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.uint8)
            mask = F.interpolate(mask.unsqueeze(0), self.target_image_size).squeeze(0) if mask.numel() > 0 else torch.empty((0, self.target_image_size[1], self.target_image_size[0]), dtype = torch.uint8)
            #img = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
            #mask = F.interpolate(img.unsqueeze(0), scale_factor = scale_factor).squeeze(0)
            bbox = [bbox[0] * scale_factor, bbox[1] * scale_factor, bbox[2] * scale_factor, bbox[3] * scale_factor]
        
        bbox = torch.tensor(bbox).unsqueeze(0)
        area = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
        iscrowd = torch.zeros(len(bbox), dtype = torch.uint8)
        labels = 1 + torch.tensor(self.category_idx[m['category']]).unsqueeze(0)
        masks = (mask == 255).unsqueeze(0)
        object_location = torch.as_tensor(m['trans_mat'], dtype = torch.float64).unsqueeze(0)
        object_rotation = torch.as_tensor(m['rot_mat'], dtype = torch.float64).unsqueeze(0)
        shape_idx = torch.tensor(self.shape_idx[m['model']]).unsqueeze(0)

        img = img / 255.0
        target = dict(
            image_id = m['img'],
            shape_path = m['model'],
            mask_path = m['mask'],
            category = m['category'], 
            
            boxes = bbox,
            area = area,
            iscrowd = iscrowd,
            labels = labels,
            masks = masks, 

            image_width_height = img_size,
            shape_idx = shape_idx,
            object_location = object_location,
            object_rotation = object_rotation
        )
        
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

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
                segmentation = pycocotools.mask.encode( torchvision.io.read_image(os.path.join(self.root, m['mask']))[0].eq(255).to(torch.uint8).t().contiguous().t().numpy() )[0],

                rot_mat = m['rot_mat'], 
                trans_mat = m['trans_mat'],
                K = [m['focal_length'] * m['img_size'][0] / 32, m['img_size'][0] / 2, m['img_size'][1] / 2],

                shape_path = m['model']

                ) for image_idx, m in enumerate(self.metadata)]
        )
        coco_dataset.createIndex()
        return coco_dataset
