import numpy as np
import copy
import time
import torch

import pycocotools.cocoeval, pycocotools.coco, pycocotools.mask 

import utils

class CocoEvaluator(object):
    def __init__(self, cocoapi, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.cocoapi = cocoapi # copy.deepcopy(cocoapi) if not RLE
        self.coco_eval = {iou_type : pycocotools.cocoeval.COCOeval(self.cocoapi, iouType = iou_type) for iou_type in iou_types}
        self.eval_imgs = {k: [] for k in iou_types}
        self.img_ids = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type, coco_eval in self.coco_eval.items():
            results = self.prepare(predictions, iou_type)
            coco_dt = self.cocoapi.loadRes(results) if results else pycocotools.coco.COCO()

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.evaluate()
            
            eval_imgs = np.asarray(coco_eval.evalImgs).reshape(-1, len(coco_eval.params.areaRng), len(coco_eval.params.imgIds))
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type, coco_eval in self.coco_eval.items():
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            
            all_img_ids = utils.all_gather(self.img_ids)
            all_eval_imgs = utils.all_gather(self.eval_imgs[iou_type])

            merged_img_ids = np.array([s for p in all_img_ids for s in p])
            merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
            merged_eval_imgs = np.concatenate(all_eval_imgs, 2)
            merged_eval_imgs = merged_eval_imgs[..., idx]

            coco_eval.evalImgs = list(merged_eval_imgs.flatten())
            coco_eval.params.imgIds = list(merged_img_ids)
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

    def evaluate(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric:", iou_type)
            coco_eval.accumulate()
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        assert iou_type in ["bbox", "segm"]
        
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
        
            if iou_type == "bbox":
                coco_results.extend(
                    dict(image_id = original_id, category_id = l, score = s, bbox = b)
                    for s, l, b in zip(prediction["scores"].tolist(), prediction["labels"].tolist(), self.xyxy_to_xywh(prediction["boxes"]).tolist())
                )
            elif iou_type == "segm":
                coco_results.extend(
                    dict(image_id = original_id, category_id = l, score = s, segmentation = r)
                    for s, l, m in zip(prediction["scores"].tolist(), prediction["labels"].tolist(), prediction["rle"])
                )

        return coco_results

    @staticmethod
    def xyxy_to_xywh(boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(dim=-1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)
