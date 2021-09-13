import numpy as np
import copy
import time
import torch

import pycocotools.cocoeval, pycocotools.coco, pycocotools.mask 

import utils

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        self.coco_gt = copy.deepcopy(coco_gt)
        self.coco_eval = {iou_type : pycocotools.cocoeval.COCOeval(self.coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.eval_imgs = {k: [] for k in iou_types}
        self.img_ids = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type, coco_eval in self.coco_eval.items():
            results = self.prepare(predictions, iou_type)
            coco_dt = self.coco_gt.loadRes(results) if results else pycocotools.coco.COCO()

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
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            boxes = self.xyxy_to_xywh(prediction["boxes"]).tolist()

            coco_results.extend(
                dict(
                    image_id = original_id,
                    category_id = labels[k],
                    bbox = box,
                    score = scores[k],
                )
                for k, box in enumerate(boxes)
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            masks = prediction["masks"] > 0.5

            rles = [
                dict(rle, counts = rle['counts'].decode('utf-8'))
                for mask in masks for rle in [pycocotools.mask.encode(mask.to(torch.uint8).t().contiguous().t().unsqueeze(-1).numpy())[0]]
            ]

            coco_results.extend(
                dict(
                    image_id = original_id,
                    category_id = labels[k],
                    segmentation = rle,
                    score = scores[k],
                )
                for k, rle in enumerate(rles)
            )
        return coco_results

    @staticmethod
    def xyxy_to_xywh(boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(dim=-1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)
