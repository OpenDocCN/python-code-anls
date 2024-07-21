# `.\pytorch\android\test_app\app\src\main\java\org\pytorch\testapp\CameraActivity.java`

```
  // Android 应用程序的摄像头活动类，继承自 AppCompatActivity
  public class CameraActivity extends AppCompatActivity {
    // 调试日志的标签，使用 BuildConfig 中的 LOGCAT_TAG
    private static final String TAG = BuildConfig.LOGCAT_TAG;
    // 文本截断大小常量
    private static final int TEXT_TRIM_SIZE = 4096;

    // 请求相机权限的请求码
    private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
    // 相机权限数组
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

    // 上次分析结果时间的长整型变量
    private long mLastAnalysisResultTime;

    // 后台线程
    protected HandlerThread mBackgroundThread;
    // 后台处理程序
    protected Handler mBackgroundHandler;
    // 主线程处理程序
    protected Handler mUIHandler;

    // 文本视图
    private TextView mTextView;
    // 文本视图字符串构建器
    private StringBuilder mTextViewStringBuilder = new StringBuilder();

    @Override
    // 在活动创建时调用
    protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      // 设置内容视图为 activity_camera.xml
      setContentView(R.layout.activity_camera);
      // 获取文本视图控件
      mTextView = findViewById(R.id.text);
      // 获取主线程的处理程序
      mUIHandler = new Handler(getMainLooper());
      // 启动后台线程
      startBackgroundThread();

      // 检查是否有相机权限，如果没有则请求
      if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
          != PackageManager.PERMISSION_GRANTED) {
        ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_CODE_CAMERA_PERMISSION);
      } else {
        // 设置 CameraX
        setupCameraX();
      }
    }

    @Override
    // 活动创建后的回调
    protected void onPostCreate(@Nullable Bundle savedInstanceState) {
      super.onPostCreate(savedInstanceState);
      // 再次启动后台线程
      startBackgroundThread();
    }

    // 启动后台线程的方法
    protected void startBackgroundThread() {
      // 创建一个名为 "ModuleActivity" 的新线程
      mBackgroundThread = new HandlerThread("ModuleActivity");
      // 启动线程
      mBackgroundThread.start();
      // 获取线程的 Looper 并创建后台处理程序
      mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    @Override
    // 销毁活动时调用
    protected void onDestroy() {
      // 停止后台线程
      stopBackgroundThread();
      super.onDestroy();
    }

    // 停止后台线程的方法
    protected void stopBackgroundThread() {
      // 安全退出后台线程
      mBackgroundThread.quitSafely();
      try {
        // 等待线程结束
        mBackgroundThread.join();
        // 线程置空
        mBackgroundThread = null;
        // 处理程序置空
        mBackgroundHandler = null;
      } catch (InterruptedException e) {
        // 输出错误日志
        Log.e(TAG, "Error on stopping background thread", e);
      }
  }
  // 结束了方法的定义块

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    // 检查请求码是否为相机权限请求码
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      // 检查权限授予情况，若拒绝授予相机权限
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        // 显示长时提示消息，提醒需要相机权限才能使用图像分类示例，并结束当前活动
        Toast.makeText(
                this,
                "You can't use image classification example without granting CAMERA permission",
                Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        // 权限已授予，初始化相机设置
        setupCameraX();
      }
    }
  }

  private static final int TENSOR_WIDTH = 224;
  private static final int TENSOR_HEIGHT = 224;

  // 设置相机预览和图像分析的方法
  private void setupCameraX() {
    // 查找相机纹理视图，并设置预览配置
    final TextureView textureView =
        ((ViewStub) findViewById(R.id.camera_texture_view_stub))
            .inflate()
            .findViewById(R.id.texture_view);
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);
    // 设置预览输出更新监听器，将预览输出设置到纹理视图中
    preview.setOnPreviewOutputUpdateListener(
        new Preview.OnPreviewOutputUpdateListener() {
          @Override
          public void onUpdated(Preview.PreviewOutput output) {
            textureView.setSurfaceTexture(output.getSurfaceTexture());
          }
        });

    // 配置图像分析的设置，包括目标分辨率、回调处理程序和图像阅读模式
    final ImageAnalysisConfig imageAnalysisConfig =
        new ImageAnalysisConfig.Builder()
            .setTargetResolution(new Size(TENSOR_WIDTH, TENSOR_HEIGHT))
            .setCallbackHandler(mBackgroundHandler)
            .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    // 设置图像分析的分析器，处理来自相机的图像
    imageAnalysis.setAnalyzer(
        new ImageAnalysis.Analyzer() {
          @Override
          public void analyze(ImageProxy image, int rotationDegrees) {
            // 检查自上次分析结果以来的时间间隔是否小于500毫秒
            if (SystemClock.elapsedRealtime() - mLastAnalysisResultTime < 500) {
              return;
            }

            // 调用分析图像方法，获取结果
            final Result result = CameraActivity.this.analyzeImage(image, rotationDegrees);

            // 如果结果不为空，更新上次分析结果时间，并在UI线程上处理结果
            if (result != null) {
              mLastAnalysisResultTime = SystemClock.elapsedRealtime();
              CameraActivity.this.runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      CameraActivity.this.handleResult(result);
                    }
                  });
            }
          }
        });

    // 将预览和图像分析绑定到生命周期管理器中
    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  private Module mModule;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;

  // 在工作线程中分析图像的方法，返回分析结果或null
  @WorkerThread
  @Nullable
  protected Result analyzeImage(ImageProxy image, int rotationDegrees) {
    // 记录分析图像的日志信息
    Log.i(TAG, String.format("analyzeImage(%s, %d)", image, rotationDegrees));
    // 如果模块为空，则加载模块
    if (mModule == null) {
      // 打印日志，指示正在从特定资源加载模块
      Log.i(TAG, "Loading module from asset '" + BuildConfig.MODULE_ASSET_NAME + "'");
      // 使用 PyTorchAndroid 从资产中加载模块，并赋给 mModule
      mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
      // 分配一个 float 类型的缓冲区，用于输入张量
      mInputTensorBuffer = Tensor.allocateFloatBuffer(3 * TENSOR_WIDTH * TENSOR_HEIGHT);
      // 创建一个输入张量，形状为 [1, 3, TENSOR_WIDTH, TENSOR_HEIGHT]
      mInputTensor =
          Tensor.fromBlob(mInputTensorBuffer, new long[] {1, 3, TENSOR_WIDTH, TENSOR_HEIGHT});
    }

    // 记录开始时间
    final long startTime = SystemClock.elapsedRealtime();
    // 将 YUV420 格式的图像中心裁剪并转换为 float 缓冲区
    TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
        image.getImage(),
        rotationDegrees,
        TENSOR_WIDTH,
        TENSOR_HEIGHT,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB,
        mInputTensorBuffer,
        0,
        MemoryFormat.CHANNELS_LAST);
    // 记录模块前向推理开始时间
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    // 执行模块的前向推理，将输入张量转换为 IValue，再将其转换为输出张量
    final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
    // 计算模块前向推理耗时
    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

    // 获取输出张量的数据作为 float 数组
    final float[] scores = outputTensor.getDataAsFloatArray();
    // 计算整体分析耗时
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

    // 返回分析结果对象
    return new Result(scores, moduleForwardDuration, analysisDuration);
  }

  // 在 UI 线程中处理分析结果
  @UiThread
  protected void handleResult(Result result) {
    // 获取分数数组中最高分的索引
    int ixs[] = Utils.topK(result.scores, 1);
    // 格式化消息字符串，包含前向推理耗时和推理结果类别
    String message =
        String.format(
            "forwardDuration:%d class:%s",
            result.moduleForwardDuration, Constants.IMAGENET_CLASSES[ixs[0]]);
    // 打印消息到日志
    Log.i(TAG, message);
    // 将消息插入到 TextView 的字符串构建器的开头
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    // 如果字符串长度超过预设的大小，进行修剪
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    // 将更新后的字符串设置为 TextView 的文本内容
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}


注释：

# 这行代码是一个单独的右花括号 '}'，用于闭合一个代码块或数据结构。
```