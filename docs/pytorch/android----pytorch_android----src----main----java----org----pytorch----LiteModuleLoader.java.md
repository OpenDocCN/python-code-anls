# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\LiteModuleLoader.java`

```py
package org.pytorch;

import android.content.res.AssetManager;
import java.util.Map;

public class LiteModuleLoader {

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on specified
   * device. The model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @param device {@link org.pytorch.Device} to use for running specified module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  // 加载序列化的TorchScript模块，从指定路径加载到特定设备上运行
  public static Module load(
      final String modelPath, final Map<String, String> extraFiles, final Device device) {
    // 创建并返回一个新的Module对象，使用LiteNativePeer处理加载细节
    return new Module(new LiteNativePeer(modelPath, extraFiles, device));
  }

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on CPU. The
   * model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  // 加载序列化的TorchScript模块，从指定路径加载到CPU上运行
  public static Module load(final String modelPath) {
    // 创建并返回一个新的Module对象，使用LiteNativePeer处理加载细节，额外文件为null，设备为CPU
    return new Module(new LiteNativePeer(modelPath, null, Device.CPU));
  }

  /**
   * Attention: This is not recommended way of loading production modules, as prepackaged assets
   * increase apk size etc. For production usage consider using loading from file on the disk {@link
   * org.pytorch.Module#load(String)}.
   *
   * <p>This method is meant to use in tests and demos.
   */
  // 注意：这种方式不推荐用于生产环境模块加载，因为预打包的资源会增加APK大小等问题。
  // 对于生产环境，请考虑使用从磁盘文件加载的方式 {@link org.pytorch.Module#load(String)}。
  // 此方法适用于测试和演示。
  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName, final Device device) {
    // 创建并返回一个新的Module对象，使用LiteNativePeer处理从资产管理器加载的细节
    return new Module(new LiteNativePeer(assetName, assetManager, device));
  }

  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName) {
    // 创建并返回一个新的Module对象，使用LiteNativePeer处理从资产管理器加载的细节，设备为CPU
    return new Module(new LiteNativePeer(assetName, assetManager, Device.CPU));
  }
}
```