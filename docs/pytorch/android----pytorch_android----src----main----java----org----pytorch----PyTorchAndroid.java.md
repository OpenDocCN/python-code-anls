# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\PyTorchAndroid.java`

```
/**
 * 导入必要的类和包，包括Android资源管理器和Facebook的JNI注解。
 */
package org.pytorch;

/**
 * PyTorchAndroid类，用于加载PyTorch模块和设置线程数。
 */
import android.content.res.AssetManager;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

/**
 * PyTorchAndroid类，负责加载PyTorch JNI库和本地库，以及配置线程数。
 */
public final class PyTorchAndroid {
  /**
   * 静态代码块，在类加载时初始化NativeLoader，并加载pytorch_jni_lite库及其它必要本地库。
   */
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("pytorch_jni_lite");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  /**
   * 从应用资源中加载模块。注意：这种方式不建议在生产环境中使用，因为预打包的资源会增加APK大小等问题。
   * 用于测试和演示目的。
   *
   * @param assetManager Android资源管理器
   * @param assetName    资源名称
   * @param device       设备类型
   * @return 加载的模块对象
   */
  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName, final Device device) {
    return new Module(new NativePeer(assetName, assetManager, device));
  }

  /**
   * 从应用资源中加载模块，默认使用CPU作为设备类型。
   *
   * @param assetManager Android资源管理器
   * @param assetName    资源名称
   * @return 加载的模块对象
   */
  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName) {
    return new Module(new NativePeer(assetName, assetManager, Device.CPU));
  }

  /**
   * 设置全局使用的本地线程数。注意：此设置对所有模块产生全局影响，使用一个线程池执行操作。
   *
   * @param numThreads 线程数，必须为正数
   * @throws IllegalArgumentException 如果线程数小于1，抛出异常
   */
  public static void setNumThreads(int numThreads) {
    if (numThreads < 1) {
      throw new IllegalArgumentException("Number of threads cannot be less than 1");
    }

    nativeSetNumThreads(numThreads);
  }

  /**
   * 本地方法声明，设置本地线程数。
   *
   * @param numThreads 线程数
   */
  @DoNotStrip
  private static native void nativeSetNumThreads(int numThreads);
}
```