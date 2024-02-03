# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\ocr_metric\icdar2015\quad_metric.py`

```
import numpy as np

from .detection.iou import DetectionIoUEvaluator


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric():
    def __init__(self, is_output_polygon=False):
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(
            is_output_polygon=is_output_polygon)
    def measure(self, batch, output, box_thresh=0.6):
        '''
        batch: (image, polygons, ignore_tags)
        batch: 由数据加载器生成的字典。
            image: 形状为 (N, C, H, W) 的张量。
            polygons: 形状为 (N, K, 4, 2) 的张量，表示目标区域的多边形。
            ignore_tags: 形状为 (N, K) 的张量，指示区域是否可忽略。
            shape: 图像的原始形状。
            filename: 图像的原始文件名。
        output: (polygons, ...)
        '''
        results = []
        gt_polyons_batch = batch['text_polys']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0])
        pred_scores_batch = np.array(output[1])
        for polygons, pred_polygons, pred_scores, ignore_tags in zip(
                gt_polyons_batch, pred_polygons_batch, pred_scores_batch,
                ignore_tags_batch):
            gt = [
                dict(
                    points=np.int64(polygons[i]), ignore=ignore_tags[i])
                for i in range(len(polygons))
            ]
            if self.is_output_polygon:
                pred = [
                    dict(points=pred_polygons[i])
                    for i in range(len(pred_polygons))
                ]
            else:
                pred = []
                # print(pred_polygons.shape)
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        # print(pred_polygons[i,:,:].tolist())
                        pred.append(
                            dict(points=pred_polygons[i, :, :].astype(np.int)))
                # pred = [dict(points=pred_polygons[i,:,:].tolist()) if pred_scores[i] >= box_thresh for i in range(pred_polygons.shape[0])]
            results.append(self.evaluator.evaluate_image(gt, pred))
        return results
    # 验证测量结果，调用 measure 方法并返回结果
    def validate_measure(self, batch, output, box_thresh=0.6):
        return self.measure(batch, output, box_thresh)

    # 评估测量结果，调用 measure 方法并返回结果以及一个从0到图像数量的列表
    def evaluate_measure(self, batch, output):
        return self.measure(batch, output), np.linspace(
            0, batch['image'].shape[0]).tolist()

    # 汇总测量结果，将原始指标列表展开为一维列表，然后调用 evaluator 的 combine_results 方法
    def gather_measure(self, raw_metrics):
        raw_metrics = [
            image_metrics
            for batch_metrics in raw_metrics for image_metrics in batch_metrics
        ]

        result = self.evaluator.combine_results(raw_metrics)

        # 初始化精度、召回率和 F1 分数的计算器
        precision = AverageMeter()
        recall = AverageMeter()
        fmeasure = AverageMeter()

        # 更新精度和召回率计算器
        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        # 计算 F1 分数
        fmeasure_score = 2 * precision.val * recall.val / (
            precision.val + recall.val + 1e-8)
        fmeasure.update(fmeasure_score)

        # 返回包含精度、召回率和 F1 分数的字典
        return {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}
```