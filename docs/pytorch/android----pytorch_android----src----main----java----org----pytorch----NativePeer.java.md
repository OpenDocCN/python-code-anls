# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\NativePeer.java`

```
// 导入所需的类和接口
import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import java.util.Map;

// NativePeer 类实现了 INativePeer 接口
class NativePeer implements INativePeer {
    
  // 静态代码块，在类加载时加载本地库 "pytorch_jni"
  static {
    NativeLoader.loadLibrary("pytorch_jni");
    // 调用 PyTorchCodegenLoader 类的方法加载本地库
    PyTorchCodegenLoader.loadNativeLibs();
  }

  // 实例变量，存储本地对象的混合数据
  private final HybridData mHybridData;

  // 初始化本地对象的方法，接收模块绝对路径、额外文件映射和设备 JNI 代码
  @DoNotStrip
  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles, int deviceJniCode);

  // 初始化本地对象的方法，接收 Android 资产名称、Android 资产管理器和设备 JNI 代码
  @DoNotStrip
  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      int deviceJniCode);

  // NativePeer 类的构造函数，使用模块绝对路径和额外文件映射初始化本地对象
  NativePeer(String moduleAbsolutePath, Map<String, String> extraFiles, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles, device.jniCode);
  }

  // NativePeer 类的构造函数，使用 Android 资产名称和 Android 资产管理器初始化本地对象
  NativePeer(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      Device device) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager, device.jniCode);
  }

  // 重置本地对象的方法
  public void resetNative() {
    mHybridData.resetNative();
  }

  // 本地方法，执行前向推断操作
  @DoNotStrip
  public native IValue forward(IValue... inputs);

  // 本地方法，执行指定方法名的操作
  @DoNotStrip
  public native IValue runMethod(String methodName, IValue... inputs);
}
```