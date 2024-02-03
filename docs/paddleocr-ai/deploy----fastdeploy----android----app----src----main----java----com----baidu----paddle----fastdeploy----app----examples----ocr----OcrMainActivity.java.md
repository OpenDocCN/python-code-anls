# `.\PaddleOCR\deploy\fastdeploy\android\app\src\main\java\com\baidu\paddle\fastdeploy\app\examples\ocr\OcrMainActivity.java`

```
package com.baidu.paddle.fastdeploy.app.examples.ocr;

import static com.baidu.paddle.fastdeploy.app.ui.Utils.decodeBitmap;
import static com.baidu.paddle.fastdeploy.app.ui.Utils.getRealPathFromURI;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget TextView;

import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.view.CameraSurfaceView;
import com.baidu.paddle.fastdeploy.app.ui.view.ResultListView;
import com.baidu.paddle.fastdeploy.app.ui.Utils;
import com.baidu.paddle.fastdeploy.app.ui.view.adapter.BaseResultAdapter;
import com.baidu.paddle.fastdeploy.app.ui.view.model.BaseResultModel;
import com.baidu.paddle.fastdeploy.pipeline.PPOCRv3;
import com.baidu.paddle.fastdeploy.vision.OCRResult;
import com.baidu.paddle.fastdeploy.vision.Visualize;
import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

// OCR 主界面类，继承自 Activity 类，实现了点击事件监听和相机纹理变化监听
public class OcrMainActivity extends Activity implements View.OnClickListener, CameraSurfaceView.OnTextureChangedListener {
    // 定义一个常量字符串 TAG，用于记录当前类的简单名称
    private static final String TAG = OcrMainActivity.class.getSimpleName();

    // 声明相机预览视图对象
    CameraSurfaceView svPreview;
    // 声明用于显示状态的文本视图对象
    TextView tvStatus;
    // 声明切换按钮对象
    ImageButton btnSwitch;
    // 声明快门按钮对象
    ImageButton btnShutter;
    // 声明设置按钮对象
    ImageButton btnSettings;
    // 声明实时识别切换按钮对象
    ImageView realtimeToggleButton;
    // 声明实时状态是否运行的布尔变量
    boolean isRealtimeStatusRunning = false;
    // 声明返回到预览界面的按钮对象
    ImageView backInPreview;
    // 声明相册选择按钮对象
    private ImageView albumSelectButton;
    // 声明相机页面视图对象
    private View cameraPageView;
    // 声明结果页面视图对象
    private ViewGroup resultPageView;
    // 声明结果图片对象
    private ImageView resultImage;
    // 声明返回到结果界面的按钮对象
    private ImageView backInResult;
    // 声明置信度滑动条对象
    private SeekBar confidenceSeekbar;
    // 声明滑动条文本对象
    private TextView seekbarText;
    // 声明结果数值变量
    private float resultNum = 1.0f;
    // 声明结果列表视图对象
    private ResultListView resultView;
    // 声明图片位图对象
    private Bitmap picBitmap;
    // 声明快门位图对象
    private Bitmap shutterBitmap;
    // 声明原始图片位图对象
    private Bitmap originPicBitmap;
    // 声明原始快门位图对象
    private Bitmap originShutterBitmap;
    // 声明是否已复制快门位图的布尔变量
    private boolean isShutterBitmapCopied = false;

    // 定义几个常量用于表示不同类型的操作
    public static final int TYPE_UNKNOWN = -1;
    public static final int BTN_SHUTTER = 0;
    public static final int ALBUM_SELECT = 1;
    public static final int REALTIME_DETECT = 2;
    private static int TYPE = REALTIME_DETECT;

    // 定义几个常量用于请求权限和选择图片
    private static final int REQUEST_PERMISSION_CODE_STORAGE = 101;
    private static final int INTENT_CODE_PICK_IMAGE = 100;
    private static final int TIME_SLEEP_INTERVAL = 50; // 毫秒

    // 记录时间消耗和帧数
    long timeElapsed = 0;
    long frameCounter = 0;

    // 手动调用 'init' 和 'release' 方法
    PPOCRv3 predictor = new PPOCRv3();

    // 声明文本数组和识别分数数组
    private String[] texts;
    private float[] recScores;
    private boolean initialized;
    // 存储识别结果的列表
    private List<BaseResultModel> results = new ArrayList<>();

    @Override
    // 在创建活动时调用
    protected void onCreate(Bundle savedInstanceState) {
        // 调用父类的 onCreate 方法
        super.onCreate(savedInstanceState);

        // 设置全屏显示
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        // 设置活动的布局
        setContentView(R.layout.ocr_activity_main);

        // 初始化所有设置项，避免由于不正确的设置而导致应用崩溃
        initSettings();

        // 检查并请求相机和写入外部存储权限
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }

        // 初始化相机预览和 UI 组件
        initView();
    }

    // 忽略 lint 检查，避免对资源 ID 的非常量引用警告
    @SuppressLint("NonConstantResourceId")
    @Override
    // 点击事件监听器，根据点击的 View ID 执行不同的操作
    public void onClick(View v) {
        // 根据点击的 View ID 进行不同的操作
        switch (v.getId()) {
            // 当点击的是切换按钮时
            case R.id.btn_switch:
                // 切换相机
                svPreview.switchCamera();
                break;
            // 当点击的是快门按钮时
            case R.id.btn_shutter:
                // 设置操作类型为快门按钮
                TYPE = BTN_SHUTTER;
                // 执行拍照操作
                shutterAndPauseCamera();
                // 清空结果视图的适配器
                resultView.setAdapter(null);
                break;
            // 当点击的是设置按钮时
            case R.id.btn_settings:
                // 启动设置页面
                startActivity(new Intent(OcrMainActivity.this, OcrSettingsActivity.class));
                break;
            // 当点击的是实时识别切换按钮时
            case R.id.realtime_toggle_btn:
                // 切换实时识别样式
                toggleRealtimeStyle();
                break;
            // 当点击的是预览页面返回按钮时
            case R.id.back_in_preview:
                // 结束当前页面
                finish();
                break;
            // 当点击的是选择图片按钮时
            case R.id.iv_select:
                // 设置操作类型为相册选择
                TYPE = ALBUM_SELECT;
                // 判断是否已授予写外部存储权限
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    // 如果之前请求过该权限但用户拒绝了请求，则返回 true
                    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE_STORAGE);
                } else {
                    // 启动图片选择器
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, INTENT_CODE_PICK_IMAGE);
                }
                // 清空结果视图的适配器
                resultView.setAdapter(null);
                break;
            // 当点击的是结果页面返回按钮时
            case R.id.back_in_result:
                // 返回上一步操作
                back();
                break;
        }
    }

    // 处理返回键事件
    @Override
    public void onBackPressed() {
        super.onBackPressed();
        // 返回上一步操作
        back();
    }
    // 返回到相机页面，隐藏结果页面，重置类型为实时检测，重置快门位图复制标志，恢复预览视图，清空结果列表，重置文本和识别分数
    private void back() {
        resultPageView.setVisibility(View.GONE);
        cameraPageView.setVisibility(View.VISIBLE);
        TYPE = REALTIME_DETECT;
        isShutterBitmapCopied = false;
        svPreview.onResume();
        results.clear();
        if (texts != null) {
            texts = null;
        }
        if (recScores != null) {
            recScores = null;
        }
    }

    // 拍照并暂停相机预览
    private void shutterAndPauseCamera() {
        // 在新线程中执行
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // 等待一段时间以确保图片已正确拍摄
                    Thread.sleep(TIME_SLEEP_INTERVAL * 10); // 500ms
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                // 在主线程中执行
                runOnUiThread(new Runnable() {
                    @SuppressLint("SetTextI18n")
                    public void run() {
                        // 暂停预览视图，隐藏相机页面，显示结果页面，设置文本框显示结果数量，设置置信度滑块进度，根据快门位图设置结果图片或显示空结果对话框
                        svPreview.onPause();
                        cameraPageView.setVisibility(View.GONE);
                        resultPageView.setVisibility(View.VISIBLE);
                        seekbarText.setText(resultNum + "");
                        confidenceSeekbar.setProgress((int) (resultNum * 100));
                        if (shutterBitmap != null && !shutterBitmap.isRecycled()) {
                            resultImage.setImageBitmap(shutterBitmap);
                        } else {
                            new AlertDialog.Builder(OcrMainActivity.this)
                                    .setTitle("Empty Result!")
                                    .setMessage("Current picture is empty, please shutting it again!")
                                    .setCancelable(true)
                                    .show();
                        }
                    }
                });

            }
        }).start();
    }
    // 从相机拍摄的 ARGB8888 格式的图像位图中复制位图数据
    private void copyBitmapFromCamera(Bitmap ARGB8888ImageBitmap) {
        // 如果已经复制过位图数据或者传入的位图为空，则直接返回
        if (isShutterBitmapCopied || ARGB8888ImageBitmap == null) {
            return;
        }
        // 如果传入的位图没有被回收
        if (!ARGB8888ImageBitmap.isRecycled()) {
            // 使用同步块确保线程安全
            synchronized (this) {
                // 复制传入的位图数据到快门位图和原始快门位图
                shutterBitmap = ARGB8888ImageBitmap.copy(Bitmap.Config.ARGB_8888, true);
                originShutterBitmap = ARGB8888ImageBitmap.copy(Bitmap.Config.ARGB_8888, true);
            }
            // 线程休眠一段时间
            SystemClock.sleep(TIME_SLEEP_INTERVAL);
            // 标记已经复制过位图数据
            isShutterBitmapCopied = true;
        }
    }

    // 处理 Activity 返回的结果
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // 如果是选择图片的请求
        if (requestCode == INTENT_CODE_PICK_IMAGE) {
            // 如果选择图片成功
            if (resultCode == Activity.RESULT_OK) {
                // 隐藏相机页面，显示结果页面
                cameraPageView.setVisibility(View.GONE);
                resultPageView.setVisibility(View.VISIBLE);
                // 设置文本显示结果数字
                seekbarText.setText(resultNum + "");
                // 设置进度条显示结果数字的百分比
                confidenceSeekbar.setProgress((int) (resultNum * 100));
                // 获取选择图片的 URI
                Uri uri = data.getData();
                // 获取 URI 对应的真实路径
                String path = getRealPathFromURI(this, uri);
                // 解码路径对应的图片为位图，限制宽高为720x1280
                picBitmap = decodeBitmap(path, 720, 1280);
                // 复制解码后的图片位图数据到原始图片位图
                originPicBitmap = picBitmap.copy(Bitmap.Config.ARGB_8888, true);
                // 在结果图片视图中显示解码后的图片位图
                resultImage.setImageBitmap(picBitmap);
            }
        }
    }
    // 切换实时状态的样式
    private void toggleRealtimeStyle() {
        // 如果实时状态正在运行
        if (isRealtimeStatusRunning) {
            // 停止实时状态
            isRealtimeStatusRunning = false;
            // 设置实时切换按钮的图像为停止状态
            realtimeToggleButton.setImageResource(R.drawable.realtime_stop_btn);
            // 设置预览视图的纹理变化监听器为当前对象
            svPreview.setOnTextureChangedListener(this);
            // 设置状态文本视图可见
            tvStatus.setVisibility(View.VISIBLE);
        } else {
            // 如果实时状态未运行
            isRealtimeStatusRunning = true;
            // 设置实时切换按钮的图像为开始状态
            realtimeToggleButton.setImageResource(R.drawable.realtime_start_btn);
            // 设置状态文本视图不可见
            tvStatus.setVisibility(View.GONE);
            // 重置快门位图未复制标志
            isShutterBitmapCopied = false;
            // 设置预览视图的纹理变化监听器为新的匿名监听器对象
            svPreview.setOnTextureChangedListener(new CameraSurfaceView.OnTextureChangedListener() {
                @Override
                public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
                    // 如果类型为BTN_SHUTTER，则复制位图从相机
                    if (TYPE == BTN_SHUTTER) {
                        copyBitmapFromCamera(ARGB8888ImageBitmap);
                    }
                    return false;
                }
            });
        }
    }

    // 当纹理变化时调用
    @Override
    public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
        // 如果类型为BTN_SHUTTER，则复制位图从相机
        if (TYPE == BTN_SHUTTER) {
            copyBitmapFromCamera(ARGB8888ImageBitmap);
            return false;
        }

        boolean modified = false;

        // 记录当前时间
        long tc = System.currentTimeMillis();
        // 预测OCR结果
        OCRResult result = predictor.predict(ARGB8888ImageBitmap);
        // 计算时间消耗
        timeElapsed += (System.currentTimeMillis() - tc);

        // 可视化OCR结果
        Visualize.visOcr(ARGB8888ImageBitmap, result);
        // 检查结果是否初始化
        modified = result.initialized();

        // 帧计数器递增
        frameCounter++;
        // 如果帧计数器大于等于30
        if (frameCounter >= 30) {
            // 计算帧率
            final int fps = (int) (1000 / (timeElapsed / 30));
            // 在UI线程更新状态文本视图显示帧率
            runOnUiThread(new Runnable() {
                @SuppressLint("SetTextI18n")
                public void run() {
                    tvStatus.setText(Integer.toString(fps) + "fps");
                }
            });
            // 重置帧计数器和时间消耗
            frameCounter = 0;
            timeElapsed = 0;
        }
        return modified;
    }

    @Override
    // 当 Activity 重新进入前台时调用
    protected void onResume() {
        super.onResume();
        // 重新加载设置并重新初始化预测器
        checkAndUpdateSettings();
        // 在权限被授予之前打开相机
        if (!checkAllPermissions()) {
            svPreview.disableCamera();
        } else {
            svPreview.enableCamera();
        }
        // 恢复预览视图
        svPreview.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        // 暂停预览视图
        svPreview.onPause();
    }

    @Override
    protected void onDestroy() {
        // 如果预测器不为空，释放资源
        if (predictor != null) {
            predictor.release();
        }
        super.onDestroy();
    }

    // 处理图像细节
    private void detail(Bitmap bitmap) {
        // 使用预测器对位图进行预测
        OCRResult result = predictor.predict(bitmap, true);

        // 获取预测结果的文本和置信度
        texts = result.mText;
        recScores = result.mRecScores;

        // 检查预测器是否已初始化
        initialized = result.initialized();
        if (initialized) {
            // 遍历文本数组，筛选出置信度高于 resultNum 的结果
            for (int i = 0; i < texts.length; i++) {
                if (recScores[i] > resultNum) {
                    results.add(new BaseResultModel(i + 1, texts[i], recScores[i]));
                }
            }
        }
        // 创建适配器并设置给结果视图
        BaseResultAdapter adapter = new BaseResultAdapter(getBaseContext(), R.layout.ocr_result_page_item, results);
        resultView.setAdapter(adapter);
        resultView.invalidate();

        // 设置结果图像和 resultNum
        resultImage.setImageBitmap(bitmap);
        resultNum = 1.0f;
    }

    // 初始化设置
    @SuppressLint("ApplySharedPref")
    public void initSettings() {
        // 获取默认的共享偏好设置
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        // 清除所有设置
        editor.clear();
        editor.commit();
        // 重置 OCR 设置
        OcrSettingsActivity.resetSettings();
    }

    @Override
    // 当权限请求结果返回时调用的方法
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        // 调用父类的方法处理权限请求结果
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // 如果权限未被授予，则显示警告对话框
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            new AlertDialog.Builder(OcrMainActivity.this)
                    .setTitle("Permission denied")
                    .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                            "App->Permissions to grant all of the permissions.")
                    .setCancelable(false)
                    // 设置强制退出按钮
                    .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            // 退出当前 Activity
                            OcrMainActivity.this.finish();
                        }
                    }).show();
        }
    }

    // 请求所有权限的方法
    private void requestAllPermissions() {
        // 请求写入外部存储和相机权限
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA}, 0);
    }

    // 检查是否已经获取了所有权限的方法
    private boolean checkAllPermissions() {
        // 检查写入外部存储和相机权限是否已经被授予
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
# 闭合之前的代码块
```