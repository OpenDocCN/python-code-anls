# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\Predictor.java`

```py
package com.baidu.paddle.lite.demo.ocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Point;
import android.util.Log;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import static android.graphics.Color.*;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    public int cpuThreadNum = 4;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String modelName = "";
    protected OCRPredictorNative paddlePredictor = null;
    protected float inferenceTime = 0;
    // Only for object detection
    protected Vector<String> wordLabels = new Vector<String>();
    protected int detLongSize = 960;
    protected float scoreThreshold = 0.1f;
    protected Bitmap inputImage = null;
    protected Bitmap outputImage = null;
    protected volatile String outputResult = "";
    protected float postprocessTime = 0;

    // 构造函数
    public Predictor() {
    }

    // 初始化方法，加载模型和标签
    public boolean init(Context appCtx, String modelPath, String labelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode) {
        // 加载模型
        isLoaded = loadModel(appCtx, modelPath, useOpencl, cpuThreadNum, cpuPowerMode);
        // 如果加载模型失败，则返回false
        if (!isLoaded) {
            return false;
        }
        // 加载标签
        isLoaded = loadLabel(appCtx, labelPath);
        return isLoaded;
    }
    // 初始化方法，传入应用上下文、模型路径、标签路径、是否使用OpenCL、CPU线程数、CPU功耗模式、检测长边大小、得分阈值
    public boolean init(Context appCtx, String modelPath, String labelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode,
                        int detLongSize, float scoreThreshold) {
        // 调用另一个初始化方法，加载模型和标签
        boolean isLoaded = init(appCtx, modelPath, labelPath, useOpencl, cpuThreadNum, cpuPowerMode);
        // 如果加载失败，则返回false
        if (!isLoaded) {
            return false;
        }
        // 设置检测长边大小和得分阈值
        this.detLongSize = detLongSize;
        this.scoreThreshold = scoreThreshold;
        // 返回true表示初始化成功
        return true;
    }
    // 加载模型的方法，传入应用上下文、模型路径、是否使用 OpenCL、CPU 线程数、CPU 功耗模式
    protected boolean loadModel(Context appCtx, String modelPath, int useOpencl, int cpuThreadNum, String cpuPowerMode) {
        // 释放已存在的模型资源
        releaseModel();

        // 加载模型
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // 如果模型路径的第一个字符不是 '/'，则从自定义路径读取模型文件，否则从 assets 复制模型到缓存
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }

        // 配置 OCRPredictorNative.Config 对象
        OCRPredictorNative.Config config = new OCRPredictorNative.Config();
        config.useOpencl = useOpencl;
        config.cpuThreadNum = cpuThreadNum;
        config.cpuPower = cpuPowerMode;
        config.detModelFilename = realPath + File.separator + "det_db.nb";
        config.recModelFilename = realPath + File.separator + "rec_crnn.nb";
        config.clsModelFilename = realPath + File.separator + "cls.nb";
        Log.i("Predictor", "model path" + config.detModelFilename + " ; " + config.recModelFilename + ";" + config.clsModelFilename);
        // 创建 OCRPredictorNative 对象
        paddlePredictor = new OCRPredictorNative(config);

        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    // 释放模型资源的方法
    public void releaseModel() {
        if (paddlePredictor != null) {
            // 销毁 paddlePredictor 对象
            paddlePredictor.destory();
            paddlePredictor = null;
        }
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }
    // 从给定路径加载标签文件，返回是否加载成功
    protected boolean loadLabel(Context appCtx, String labelPath) {
        // 清空 wordLabels 列表
        wordLabels.clear();
        // 添加默认标签 "black"
        wordLabels.add("black");
        // 从文件中加载词标签
        try {
            // 打开 assets 文件夹中的文件流
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            // 获取文件流可读取的字节数
            int available = assetsInputStream.available();
            // 创建字节数组存储文件内容
            byte[] lines = new byte[available];
            // 读取文件内容到字节数组
            assetsInputStream.read(lines);
            // 关闭文件流
            assetsInputStream.close();
            // 将字节数组转换为字符串
            String words = new String(lines);
            // 根据换行符分割字符串，获取词标签数组
            String[] contents = words.split("\n");
            // 将词标签添加到 wordLabels 列表中
            for (String content : contents) {
                wordLabels.add(content);
            }
            // 添加空格标签
            wordLabels.add(" ");
            // 打印词标签数量
            Log.i(TAG, "Word label size: " + wordLabels.size());
        } catch (Exception e) {
            // 打印异常信息
            Log.e(TAG, e.getMessage());
            // 返回加载失败
            return false;
        }
        // 返回加载成功
        return true;
    }

    // 运行模型进行推理
    public boolean runModel(int run_det, int run_cls, int run_rec) {
        // 检查输入图像和模型是否加载
        if (inputImage == null || !isLoaded()) {
            return false;
        }

        // 预热模型
        for (int i = 0; i < warmupIterNum; i++) {
            paddlePredictor.runImage(inputImage, detLongSize, run_det, run_cls, run_rec);
        }
        // 取消预热
        warmupIterNum = 0;
        // 运行推理
        Date start = new Date();
        ArrayList<OcrResultModel> results = paddlePredictor.runImage(inputImage, detLongSize, run_det, run_cls, run_rec);
        Date end = new Date();
        // 计算推理时间
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;

        // 后处理结果
        results = postprocess(results);
        // 打印推理时间和结果框数量
        Log.i(TAG, "[stat] Inference Time: " + inferenceTime + " ;Box Size " + results.size());
        // 绘制结果
        drawResults(results);

        // 返回运行成功
        return true;
    }

    // 检查模型是否加载
    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    // 返回模型路径
    public String modelPath() {
        return modelPath;
    }

    // 返回模型名称
    public String modelName() {
        return modelName;
    }
    // 返回 CPU 线程数
    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    // 返回 CPU 功耗模式
    public String cpuPowerMode() {
        return cpuPowerMode;
    }

    // 返回推理时间
    public float inferenceTime() {
        return inferenceTime;
    }

    // 返回输入图像
    public Bitmap inputImage() {
        return inputImage;
    }

    // 返回输出图像
    public Bitmap outputImage() {
        return outputImage;
    }

    // 返回输出结果
    public String outputResult() {
        return outputResult;
    }

    // 返回后处理时间
    public float postprocessTime() {
        return postprocessTime;
    }

    // 设置输入图像
    public void setInputImage(Bitmap image) {
        // 如果输入图像为空，则返回
        if (image == null) {
            return;
        }
        // 复制输入图像并设置配置为 ARGB_8888
        this.inputImage = image.copy(Bitmap.Config.ARGB_8888, true);
    }

    // 后处理方法，处理识别结果
    private ArrayList<OcrResultModel> postprocess(ArrayList<OcrResultModel> results) {
        // 遍历识别结果
        for (OcrResultModel r : results) {
            // 创建一个 StringBuffer 用于存储识别结果的文字
            StringBuffer word = new StringBuffer();
            // 遍历识别结果的单词索引
            for (int index : r.getWordIndex()) {
                // 如果索引在标签列表范围内，则将对应的标签添加到 word 中，否则添加 ×
                if (index >= 0 && index < wordLabels.size()) {
                    word.append(wordLabels.get(index));
                } else {
                    // 输出错误日志
                    Log.e(TAG, "Word index is not in label list:" + index);
                    word.append("×");
                }
            }
            // 设置识别结果的标签为拼接后的文字
            r.setLabel(word.toString());
            // 设置识别结果的类别标签
            r.setClsLabel(r.getClsIdx() == 1 ? "180" : "0");
        }
        // 返回处理后的识别结果列表
        return results;
    }
# 闭合之前的代码块
```