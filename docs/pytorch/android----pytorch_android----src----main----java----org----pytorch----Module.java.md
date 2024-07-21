# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\Module.java`

```
// Copyright 2004-present Facebook. All Rights Reserved.

// 引入必要的依赖和类
package org.pytorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/** Java wrapper for torch::jit::Module. */
// 定义 Module 类，用于封装 torch::jit::Module
public class Module {

  private INativePeer mNativePeer;

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on specified
   * device.
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @param device {@link org.pytorch.Device} to use for running specified module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::Module.
   */
  // 静态方法 load，加载指定路径上的序列化 TorchScript 模块，可以指定设备
  public static Module load(
      final String modelPath, final Map<String, String> extraFiles, final Device device) {
    // 如果 NativeLoader 没有初始化，进行初始化
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // 返回一个新的 Module 对象，包含加载后的 NativePeer 对象
    return new Module(new NativePeer(modelPath, extraFiles, device));
  }

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::Module.
   */
  // 另一个静态方法 load，加载指定路径上的序列化 TorchScript 模块，运行在 CPU 上
  public static Module load(final String modelPath) {
    // 调用 load 方法，默认使用 CPU 设备
    return load(modelPath, null, Device.CPU);
  }

  // Module 类的构造函数，接受一个 INativePeer 对象作为参数
  Module(INativePeer nativePeer) {
    this.mNativePeer = nativePeer;
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the TorchScript module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  // 调用 NativePeer 对象的 forward 方法，运行 TorchScript 模块的 forward 方法
  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the TorchScript method to run.
   * @param inputs arguments that will be passed to TorchScript method.
   * @return return value from the method.
   */
  // 调用 NativePeer 对象的 runMethod 方法，运行指定的 TorchScript 方法
  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  /**
   * Explicitly destroys the native torch::jit::Module. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  // 显式销毁 native torch::jit::Module，尽管垃圾回收时会自动销毁，但调用 destroy 方法可以更快地释放内存
  public void destroy() {
    mNativePeer.resetNative();
  }
}
```