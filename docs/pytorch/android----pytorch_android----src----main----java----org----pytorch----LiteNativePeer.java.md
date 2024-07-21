# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\LiteNativePeer.java`

```py
package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/**
 * LiteNativePeer类实现了INativePeer接口。
 */
class LiteNativePeer implements INativePeer {

  static {
    // 检查NativeLoader是否已初始化，如果没有，则使用SystemDelegate进行初始化
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // 加载名为"pytorch_jni_lite"的本地库
    NativeLoader.loadLibrary("pytorch_jni_lite");
    // 加载PyTorch代码生成器相关的本地库
    PyTorchCodegenLoader.loadNativeLibs();
  }

  private final HybridData mHybridData;

  /**
   * 使用给定的模块路径、额外文件和设备JNI代码初始化HybridData对象。
   *
   * @param moduleAbsolutePath 模块的绝对路径
   * @param extraFiles 额外文件的映射，文件名到内容的映射
   * @param device 设备对象，包含JNI代码
   */
  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles, int deviceJniCode);

  /**
   * 使用给定的资产名称、Android资产管理器和设备JNI代码初始化HybridData对象。
   *
   * @param assetName 资产的名称
   * @param androidAssetManager Android资产管理器对象
   * @param device 设备对象，包含JNI代码
   */
  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */ Object androidAssetManager,
      int deviceJniCode);

  /**
   * LiteNativePeer的构造函数，使用模块路径、额外文件映射和设备对象创建实例。
   *
   * @param moduleAbsolutePath 模块的绝对路径
   * @param extraFiles 额外文件的映射，文件名到内容的映射
   * @param device 设备对象，包含JNI代码
   */
  LiteNativePeer(String moduleAbsolutePath, Map<String, String> extraFiles, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles, device.jniCode);
  }

  /**
   * LiteNativePeer的构造函数，使用资产名称、Android资产管理器对象和设备对象创建实例。
   *
   * @param assetName 资产的名称
   * @param androidAssetManager Android资产管理器对象
   * @param device 设备对象，包含JNI代码
   */
  LiteNativePeer(String assetName, /* android.content.res.AssetManager */ Object androidAssetManager,
      Device device) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager, device.jniCode);
  }

  /**
   * 显式销毁native torch::jit::mobile::Module。调用此方法并非必须，因为当对象被垃圾回收时，
   * 本地对象会被销毁。但不能保证垃圾回收的时机，因此主动调用resetNative方法可以更快地释放内存。
   * 参见 {@link com.facebook.jni.HybridData#resetNative}。
   */
  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * 运行该模块的'forward'方法，并传入指定的参数。
   *
   * @param inputs 传递给TorchScript模块'forward'方法的参数
   * @return 'forward'方法的返回值
   */
  public native IValue forward(IValue... inputs);

  /**
   * 运行该模块的指定方法，并传入指定的参数。
   *
   * @param methodName 要运行的TorchScript方法的名称
   * @param inputs 将传递给TorchScript方法的参数
   * @return 方法的返回值
   */
  public native IValue runMethod(String methodName, IValue... inputs);
}
```