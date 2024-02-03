# `.\PaddleOCR\deploy\android_demo\app\src\main\java\com\baidu\paddle\lite\demo\ocr\MainActivity.java`

```py
# 导入必要的包
package com.baidu.paddle.lite.demo.ocr;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.content.res.AssetManager;
import android.media.FaceDetector;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.io InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

# 定义 MainActivity 类，继承自 AppCompatActivity
public class MainActivity extends AppCompatActivity {
    # 定义 TAG 常量，用于日志输出
    private static final String TAG = MainActivity.class.getSimpleName();
    # 定义打开相册请求码和拍照请求码
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;

    # 定义加载模型和运行模型请求码，以及加载模型成功和失败的响应码
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    // 定义运行模型成功和失败的响应常量
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    // 加载模型时显示的进度对话框
    protected ProgressDialog pbLoadModel = null;
    // 运行模型时显示的进度对话框
    protected ProgressDialog pbRunModel = null;

    // 用于接收来自工作线程的消息的处理程序
    protected Handler receiver = null;
    // 用于向工作线程发送命令的处理程序
    protected Handler sender = null;
    // 加载和运行模型的工作线程
    protected HandlerThread worker = null;

    // 对象检测的 UI 组件
    protected TextView tvInputSetting;
    protected TextView tvStatus;
    protected ImageView ivInputImage;
    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;
    protected CheckBox cbOpencl;
    protected Spinner spRunMode;

    // OCR 模型设置
    protected String modelPath = "";
    protected String labelPath = "";
    protected String imagePath = "";
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "";
    protected int detLongSize = 960;
    protected float scoreThreshold = 0.1f;
    private String currentPhotoPath;
    private AssetManager assetManager = null;

    // 预测器对象
    protected Predictor predictor = new Predictor();

    // 当前预测的图像
    private Bitmap cur_predict_image = null;

    @Override
    }

    @Override
    }

    // 加载模型的方法
    public void loadModel() {
        // 显示加载模型的进度对话框
        pbLoadModel = ProgressDialog.show(this, "", "loading model...", false, false);
        // 向工作线程发送加载模型的请求消息
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    // 运行模型的方法
    public void runModel() {
        // 显示运行模型的进度对话框
        pbRunModel = ProgressDialog.show(this, "", "running model...", false, false);
        // 向工作线程发送运行模型的请求消息
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    // 加载模型的逻辑
    public boolean onLoadModel() {
        // 如果预测器已加载模型，则释放模型
        if (predictor.isLoaded()) {
            predictor.releaseModel();
        }
        // 初始化预测器，传入相关参数
        return predictor.init(MainActivity.this, modelPath, labelPath, cbOpencl.isChecked() ? 1 : 0, cpuThreadNum,
                cpuPowerMode,
                detLongSize, scoreThreshold);
    }
    // 根据选择的运行模式确定检测、分类、识别是否需要运行
    public boolean onRunModel() {
        // 获取选择的运行模式
        String run_mode = spRunMode.getSelectedItem().toString();
        // 判断是否需要运行检测
        int run_det = run_mode.contains("检测") ? 1 : 0;
        // 判断是否需要运行分类
        int run_cls = run_mode.contains("分类") ? 1 : 0;
        // 判断是否需要运行识别
        int run_rec = run_mode.contains("识别") ? 1 : 0;
        // 返回模型是否加载成功并且运行成功
        return predictor.isLoaded() && predictor.runModel(run_det, run_cls, run_rec);
    }

    // 加载模型成功时的操作
    public void onLoadModelSuccessed() {
        // 显示模型信息和设置信息
        tvInputSetting.setText("Model: " + modelPath.substring(modelPath.lastIndexOf("/") + 1) + "\nOPENCL: " + cbOpencl.isChecked() + "\nCPU Thread Num: " + cpuThreadNum + "\nCPU Power Mode: " + cpuPowerMode);
        tvInputSetting.scrollTo(0, 0);
        // 显示加载模型成功的状态
        tvStatus.setText("STATUS: load model successed");
    }

    // 加载模型失败时的操作
    public void onLoadModelFailed() {
        // 显示加载模型失败的状态
        tvStatus.setText("STATUS: load model failed");
    }

    // 运行模型成功时的操作
    public void onRunModelSuccessed() {
        // 显示运行模型成功的状态
        tvStatus.setText("STATUS: run model successed");
        // 显示推理时间
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        // 获取输出图像并显示
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        // 显示输出结果
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }

    // 运行模型失败时的操作
    public void onRunModelFailed() {
        // 显示运行模型失败的状态
        tvStatus.setText("STATUS: run model failed");
    }

    // 设置图像的操作
    public void set_img() {
        // 从路径加载测试图像并运行模型
        try {
            // 获取 AssetsManager
            assetManager = getAssets();
            // 打开输入流
            InputStream in = assetManager.open(imagePath);
            // 解码输入流为 Bitmap
            Bitmap bmp = BitmapFactory.decodeStream(in);
            // 设置当前预测图像
            cur_predict_image = bmp;
            // 在 ImageView 中显示图像
            ivInputImage.setImageBitmap(bmp);
        } catch (IOException e) {
            // 加载图像失败时显示提示信息
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            // 打印异常信息
            e.printStackTrace();
        }
    }
    // 当设置按钮被点击时，启动设置活动
    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    // 创建选项菜单
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    // 准备选项菜单
    public boolean onPrepareOptionsMenu(Menu menu) {
        // 检查预测器是否已加载
        boolean isLoaded = predictor.isLoaded();
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    // 处理选项菜单项被选择的事件
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                // 返回上一个活动
                finish();
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // 确保我们有SD卡读写权限以从SD卡加载模型
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    // 处理权限请求结果
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // 如果权限被拒绝，则显示提示信息
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }
    // 请求所有必要的权限
    private boolean requestAllPermissions() {
        // 检查是否缺少写外部存储和相机权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            // 请求写外部存储和相机权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        // 已经拥有权限，返回true
        return true;
    }

    // 打开相册
    private void openGallery() {
        // 创建一个意图，用于选择图片
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        // 设置意图的数据和类型为外部存储中的图片
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        // 启动选择图片的活动，并等待结果
        startActivityForResult(intent, OPEN_GALLERY_REQUEST_CODE);
    }
    // 拍照功能的方法
    private void takePhoto() {
        // 创建一个用于拍照的Intent
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // 确保有相机应用可以处理该Intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // 创建存储照片的文件
            File photoFile = null;
            try {
                // 尝试创建照片文件
                photoFile = createImageFile();
            } catch (IOException ex) {
                // 如果创建文件失败，记录错误信息并显示Toast提示
                Log.e("MainActitity", ex.getMessage(), ex);
                Toast.makeText(MainActivity.this,
                        "Create Camera temp file failed: " + ex.getMessage(), Toast.LENGTH_SHORT).show();
            }
            // 只有成功创建文件才继续
            if (photoFile != null) {
                // 获取存储图片的路径并创建对应的URI
                Log.i(TAG, "FILEPATH " + getExternalFilesDir("Pictures").getAbsolutePath());
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.baidu.paddle.lite.demo.ocr.fileprovider",
                        photoFile);
                currentPhotoPath = photoFile.getAbsolutePath();
                // 将照片存储路径添加到Intent中，并启动拍照Activity
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, TAKE_PHOTO_REQUEST_CODE);
                Log.i(TAG, "startActivityForResult finished");
            }
        }
    }

    // 创建图片文件的方法
    private File createImageFile() throws IOException {
        // 创建图片文件名
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        // 获取存储图片的目录，并创建临时文件
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* 前缀 */
                ".bmp",         /* 后缀 */
                storageDir      /* 目录 */
        );

        return image;
    }

    // 覆盖父类的方法
    @Override
    // 当重置按钮被点击时，将当前预测的图像设置为输入图像
    public void btn_reset_img_click(View view) {
        ivInputImage.setImageBitmap(cur_predict_image);
    }

    // 当 OpenCL 复选框被点击时，设置状态文本为加载模型，并加载模型
    public void cb_opencl_click(View view) {
        tvStatus.setText("STATUS: load model ......");
        loadModel();
    }

    // 当运行模型按钮被点击时，获取输入图像，如果图像不存在则显示相应状态，否则设置状态文本为运行模型，并运行模型
    public void btn_run_model_click(View view) {
        Bitmap image = ((BitmapDrawable) ivInputImage.getDrawable()).getBitmap();
        if (image == null) {
            tvStatus.setText("STATUS: image is not exists");
        } else if (!predictor.isLoaded()) {
            tvStatus.setText("STATUS: model is not loaded");
        } else {
            tvStatus.setText("STATUS: run model ...... ");
            predictor.setInputImage(image);
            runModel();
        }
    }

    // 当选择图像按钮被点击时，如果获取到所有权限则打开相册
    public void btn_choice_img_click(View view) {
        if (requestAllPermissions()) {
            openGallery();
        }
    }

    // 当拍照按钮被点击时，如果获取到所有权限则拍照
    public void btn_take_photo_click(View view) {
        if (requestAllPermissions()) {
            takePhoto();
        }
    }

    // 在 Activity 销毁时，释放模型资源并终止 worker 线程
    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
    }

    // 获取运行模式的整数值
    public int get_run_mode() {
        String run_mode = spRunMode.getSelectedItem().toString();
        int mode;
        switch (run_mode) {
            case "检测+分类+识别":
                mode = 1;
                break;
            case "检测+识别":
                mode = 2;
                break;
            case "识别+分类":
                mode = 3;
                break;
            case "检测":
                mode = 4;
                break;
            case "识别":
                mode = 5;
                break;
            case "分类":
                mode = 6;
                break;
            default:
                mode = 1;
        }
        return mode;
    }
# 闭合之前的代码块
```