# `.\pytorch\android\test_app\app\src\main\java\org\pytorch\testapp\MainActivity.java`

```py
    // 导入所需的 Android 和 PyTorch 库
    package org.pytorch.testapp;

    import android.content.Context;
    import android.os.Bundle;
    import android.os.Handler;
    import android.os.HandlerThread;
    import android.os.SystemClock;
    import android.util.Log;
    import android.widget.TextView;
    import androidx.annotation.Nullable;
    import androidx.annotation.UiThread;
    import androidx.annotation.WorkerThread;
    import androidx.appcompat.app.AppCompatActivity;
    import java.io.File;
    import java.io.FileOutputStream;
    import java.io.IOException;
    import java.io.InputStream;
    import java.io.OutputStream;
    import java.nio.FloatBuffer;
    import org.pytorch.Device;
    import org.pytorch.IValue;
    import org.pytorch.MemoryFormat;
    import org.pytorch.Module;
    import org.pytorch.PyTorchAndroid;
    import org.pytorch.Tensor;

    public class MainActivity extends AppCompatActivity {

      private static final String TAG = BuildConfig.LOGCAT_TAG;
      private static final int TEXT_TRIM_SIZE = 4096;

      private TextView mTextView;

      // 后台线程和处理程序变量声明
      protected HandlerThread mBackgroundThread;
      protected Handler mBackgroundHandler;
      private Module mModule;
      private FloatBuffer mInputTensorBuffer;
      private Tensor mInputTensor;
      private StringBuilder mTextViewStringBuilder = new StringBuilder();

      // 执行模型前向推断的可运行任务定义
      private final Runnable mModuleForwardRunnable =
          new Runnable() {
            @Override
            public void run() {
              // 执行模型前向推断并返回结果
              final Result result = doModuleForward();
              // 在 UI 线程处理结果
              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      // 处理模型推断的结果
                      handleResult(result);
                      // 如果后台处理程序不为空，继续提交前向推断任务到后台处理程序
                      if (mBackgroundHandler != null) {
                        mBackgroundHandler.post(mModuleForwardRunnable);
                      }
                    }
                  });
            }
          };

      // 从 assets 目录复制文件到应用私有目录并返回文件路径
      public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
          return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
          try (OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
              os.write(buffer, 0, read);
            }
            os.flush();
          }
          return file.getAbsolutePath();
        } catch (IOException e) {
          // 处理异常，记录错误日志
          Log.e(TAG, "Error process asset " + assetName + " to file path");
        }
        return null;
      }

      @Override
      protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 如果是原生构建，则加载和前向模型
        if (BuildConfig.NATIVE_BUILD) {
          final String modelFileAbsoluteFilePath =
              new File(assetFilePath(this, BuildConfig.MODULE_ASSET_NAME)).getAbsolutePath();
          LibtorchNativeClient.loadAndForwardModel(modelFileAbsoluteFilePath);
          return;
        }
        // 设置布局
        setContentView(R.layout.activity_main);
        // 获取 TextView 实例
        mTextView = findViewById(R.id.text);
        // 启动后台线程处理程序
        startBackgroundThread();
      }
    # 将 mModuleForwardRunnable 对象发布到后台处理程序 mBackgroundHandler 上执行
    mBackgroundHandler.post(mModuleForwardRunnable);
  }

  # 启动后台线程
  protected void startBackgroundThread() {
    # 创建带有特定标签的 HandlerThread 对象
    mBackgroundThread = new HandlerThread(TAG + "_bg");
    # 启动后台线程
    mBackgroundThread.start();
    # 创建与后台线程关联的 Handler 对象
    mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
  }

  @Override
  protected void onDestroy() {
    # 停止后台线程的执行
    stopBackgroundThread();
    # 调用父类的 onDestroy 方法
    super.onDestroy();
  }

  # 停止后台线程的执行
  protected void stopBackgroundThread() {
    # 安全地退出后台线程
    mBackgroundThread.quitSafely();
    try {
      # 等待后台线程结束
      mBackgroundThread.join();
      # 将线程和处理程序对象置空
      mBackgroundThread = null;
      mBackgroundHandler = null;
    } catch (InterruptedException e) {
      # 捕获可能发生的中断异常
      Log.e(TAG, "Error stopping background thread", e);
    }
  }

  @WorkerThread
  @Nullable
  protected Result doModuleForward() {
    # 如果 mModule 为 null，则初始化模块及其相关属性
    if (mModule == null) {
      # 获取输入张量的形状并计算元素总数
      final long[] shape = BuildConfig.INPUT_TENSOR_SHAPE;
      long numElements = 1;
      for (int i = 0; i < shape.length; i++) {
        numElements *= shape[i];
      }
      # 分配浮点缓冲区并创建输入张量对象
      mInputTensorBuffer = Tensor.allocateFloatBuffer((int) numElements);
      mInputTensor =
          Tensor.fromBlob(
              mInputTensorBuffer, BuildConfig.INPUT_TENSOR_SHAPE, MemoryFormat.CHANNELS_LAST);
      # 设置 PyTorchAndroid 使用的线程数为 1
      PyTorchAndroid.setNumThreads(1);
      # 根据配置加载 PyTorch 模块，可能使用 Vulkan 设备加速
      mModule =
          BuildConfig.USE_VULKAN_DEVICE
              ? PyTorchAndroid.loadModuleFromAsset(
                  getAssets(), BuildConfig.MODULE_ASSET_NAME, Device.VULKAN)
              : PyTorchAndroid.loadModuleFromAsset(getAssets(), BuildConfig.MODULE_ASSET_NAME);
    }

    # 记录开始执行前向推断的时间
    final long startTime = SystemClock.elapsedRealtime();
    final long moduleForwardStartTime = SystemClock.elapsedRealtime();
    # 执行模块的前向推断并获取输出张量
    final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
    # 计算前向推断过程的持续时间
    final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
    # 将输出张量转换为 float 数组
    final float[] scores = outputTensor.getDataAsFloatArray();
    # 计算整体分析的持续时间
    final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
    # 返回包含结果数据的 Result 对象
    return new Result(scores, moduleForwardDuration, analysisDuration);
  }

  static class Result {
    # 内部类，用于存储执行结果的数据结构

    private final float[] scores;  # 分数数组
    private final long totalDuration;  # 总持续时间
    private final long moduleForwardDuration;  # 模块前向推断的持续时间

    # 结果对象的构造函数
    public Result(float[] scores, long moduleForwardDuration, long totalDuration) {
      this.scores = scores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.totalDuration = totalDuration;
    }
  }

  @UiThread
  protected void handleResult(Result result) {
    # 处理模块前向推断的结果数据
    String message = String.format("forwardDuration:%d", result.moduleForwardDuration);
    # 将消息插入到文本视图字符串构建器的开头
    mTextViewStringBuilder.insert(0, '\n').insert(0, message);
    # 如果文本长度超过指定大小，则删除多余部分
    if (mTextViewStringBuilder.length() > TEXT_TRIM_SIZE) {
      mTextViewStringBuilder.delete(TEXT_TRIM_SIZE, mTextViewStringBuilder.length());
    }
    # 更新文本视图的显示内容
    mTextView.setText(mTextViewStringBuilder.toString());
  }
}


注释：


# 这行代码表示一个代码块的结束，闭合了一个函数、循环或条件语句的范围。
# 在此处，它表明前面的代码段的执行范围已经结束，可能是一个函数或者其他语句块的结尾。
# 注意：这里没有实际的代码，仅仅是一个示例，用于展示如何对一个代码块的结束进行注释。
```