# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\OcrResultModel.java`

```
// 定义一个名为 OcrResultModel 的类，用于存储 OCR 结果
package com.baidu.paddle.lite.demo.ocr;

import android.graphics.Point;

import java.util.ArrayList;
import java.util.List;

public class OcrResultModel {
    // 声明私有成员变量 points，用于存储识别结果的坐标点
    private List<Point> points;
    // 声明私有成员变量 wordIndex，用于存储识别结果的单词索引
    private List<Integer> wordIndex;
    // 声明私有成员变量 label，用于存储识别结果的标签
    private String label;
    // 声明私有成员变量 confidence，用于存储识别结果的置信度
    private float confidence;
    // 声明私有成员变量 cls_idx，用于存储识别结果的类别索引
    private float cls_idx;
    // 声明私有成员变量 cls_label，用于存储识别结果的类别标签
    private String cls_label;
    // 声明私有成员变量 cls_confidence，用于存储识别结果的类别置信度

    public OcrResultModel() {
        super();
        // 初始化 points 和 wordIndex 为新的 ArrayList
        points = new ArrayList<>();
        wordIndex = new ArrayList<>();
    }

    // 添加一个坐标点到 points 列表中
    public void addPoints(int x, int y) {
        // 创建一个新的 Point 对象，并添加到 points 列表中
        Point point = new Point(x, y);
        points.add(point);
    }

    // 添加一个单词索引到 wordIndex 列表中
    public void addWordIndex(int index) {
        // 将单词索引添加到 wordIndex 列表中
        wordIndex.add(index);
    }

    // 返回 points 列表
    public List<Point> getPoints() {
        return points;
    }

    // 返回 wordIndex 列表
    public List<Integer> getWordIndex() {
        return wordIndex;
    }

    // 返回 label
    public String getLabel() {
        return label;
    }

    // 设置 label
    public void setLabel(String label) {
        this.label = label;
    }

    // 返回 confidence
    public float getConfidence() {
        return confidence;
    }

    // 设置 confidence
    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    // 返回 cls_idx
    public float getClsIdx() {
        return cls_idx;
    }

    // 设置 cls_idx
    public void setClsIdx(float idx) {
        this.cls_idx = idx;
    }

    // 返回 cls_label
    public String getClsLabel() {
        return cls_label;
    }

    // 设置 cls_label
    public void setClsLabel(String label) {
        this.cls_label = label;
    }

    // 返回 cls_confidence
    public float getClsConfidence() {
        return cls_confidence;
    }

    // 设置 cls_confidence
    public void setClsConfidence(float confidence) {
        this.cls_confidence = confidence;
    }
}
```