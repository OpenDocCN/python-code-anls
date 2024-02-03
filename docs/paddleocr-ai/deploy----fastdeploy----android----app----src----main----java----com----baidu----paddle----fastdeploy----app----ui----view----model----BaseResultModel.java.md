# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\ui\view\model\BaseResultModel.java`

```py
// 定义一个名为 BaseResultModel 的类，用于存储结果数据
package com.baidu.paddle.fastdeploy.app.ui.view.model;

public class BaseResultModel {
    // 定义私有整型变量 index，用于存储索引值
    private int index;
    // 定义私有字符串变量 name，用于存储名称
    private String name;
    // 定义私有浮点型变量 confidence，用于存储置信度
    private float confidence;

    // 默认构造函数，无参数
    public BaseResultModel() {

    }

    // 带参数的构造函数，用于初始化 index、name 和 confidence
    public BaseResultModel(int index, String name, float confidence) {
        this.index = index;
        this.name = name;
        this.confidence = confidence;
    }

    // 获取置信度的方法
    public float getConfidence() {
        return confidence;
    }

    // 设置置信度的方法
    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    // 获取索引值的方法
    public int getIndex() {
        return index;
    }

    // 设置索引值的方法
    public void setIndex(int index) {
        this.index = index;
    }

    // 获取名称的方法
    public String getName() {
        return name;
    }

    // 设置名称的方法
    public void setName(String name) {
        this.name = name;
    }
}
```