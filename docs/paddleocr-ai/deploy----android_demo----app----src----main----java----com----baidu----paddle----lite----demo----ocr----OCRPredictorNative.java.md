# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\OCRPredictorNative.java`

```
package com.baidu.paddle.lite.demo.ocr;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

public class OCRPredictorNative {

    private static final AtomicBoolean isSOLoaded = new AtomicBoolean();

    // 加载本地库
    public static void loadLibrary() throws RuntimeException {
        // 如果本地库未加载且成功设置为已加载状态
        if (!isSOLoaded.get() && isSOLoaded.compareAndSet(false, true)) {
            try {
                // 加载本地库
                System.loadLibrary("Native");
            } catch (Throwable e) {
                // 抛出异常
                RuntimeException exception = new RuntimeException(
                        "Load libNative.so failed, please check it exists in apk file.", e);
                throw exception;
            }
        }
    }

    private Config config;

    private long nativePointer = 0;

    // 构造函数
    public OCRPredictorNative(Config config) {
        this.config = config;
        // 加载本地库
        loadLibrary();
        // 初始化本地指针
        nativePointer = init(config.detModelFilename, config.recModelFilename, config.clsModelFilename, config.useOpencl,
                config.cpuThreadNum, config.cpuPower);
        Log.i("OCRPredictorNative", "load success " + nativePointer);

    }

    // 运行图像识别
    public ArrayList<OcrResultModel> runImage(Bitmap originalImage, int max_size_len, int run_det, int run_cls, int run_rec) {
        Log.i("OCRPredictorNative", "begin to run image ");
        // 进行前向推理
        float[] rawResults = forward(nativePointer, originalImage, max_size_len, run_det, run_cls, run_rec);
        // 后处理结果
        ArrayList<OcrResultModel> results = postprocess(rawResults);
        return results;
    }

    // 配置类
    public static class Config {
        public int useOpencl;
        public int cpuThreadNum;
        public String cpuPower;
        public String detModelFilename;
        public String recModelFilename;
        public String clsModelFilename;

    }

    // 销毁函数
    public void destory() {
        // 如果本地指针不为0
        if (nativePointer != 0) {
            // 释放资源
            release(nativePointer);
            nativePointer = 0;
        }
    }
}
    // 初始化 OCR 模型，传入检测模型路径、识别模型路径、分类模型路径、是否使用 OpenCL、线程数、CPU 模式
    protected native long init(String detModelPath, String recModelPath, String clsModelPath, int useOpencl, int threadNum, String cpuMode);

    // 对输入的 Bitmap 图像进行前向推理，返回识别结果
    protected native float[] forward(long pointer, Bitmap originalImage,int max_size_len, int run_det, int run_cls, int run_rec);

    // 释放 OCR 模型资源
    protected native void release(long pointer);

    // 对原始推理结果进行后处理，返回识别结果列表
    private ArrayList<OcrResultModel> postprocess(float[] raw) {
        // 初始化结果列表
        ArrayList<OcrResultModel> results = new ArrayList<OcrResultModel>();
        // 初始化起始位置
        int begin = 0;

        // 循环处理原始结果
        while (begin < raw.length) {
            // 获取点的数量和词的数量
            int point_num = Math.round(raw[begin]);
            int word_num = Math.round(raw[begin + 1]);
            // 解析结果并添加到结果列表
            OcrResultModel res = parse(raw, begin + 2, point_num, word_num);
            begin += 2 + 1 + point_num * 2 + word_num + 2;
            results.add(res);
        }

        // 返回结果列表
        return results;
    }

    // 解析原始结果，返回识别结果对象
    private OcrResultModel parse(float[] raw, int begin, int pointNum, int wordNum) {
        // 初始化当前位置
        int current = begin;
        // 初始化识别结果对象
        OcrResultModel res = new OcrResultModel();
        // 设置置信度
        res.setConfidence(raw[current]);
        current++;
        // 添加点坐标
        for (int i = 0; i < pointNum; i++) {
            res.addPoints(Math.round(raw[current + i * 2]), Math.round(raw[current + i * 2 + 1]));
        }
        current += (pointNum * 2);
        // 添加词的索引
        for (int i = 0; i < wordNum; i++) {
            int index = Math.round(raw[current + i]);
            res.addWordIndex(index);
        }
        current += wordNum;
        // 设置分类索引和分类置信度
        res.setClsIdx(raw[current]);
        res.setClsConfidence(raw[current + 1]);
        // 打印日志
        Log.i("OCRPredictorNative", "word finished " + wordNum);
        // 返回识别结果对象
        return res;
    }
# 闭合之前的代码块
```