# `.\pytorch\android\test_app\app\src\main\java\org\pytorch\testapp\LibtorchNativeClient.java`

```py
package org.pytorch.testapp;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class LibtorchNativeClient {

  // 加载并前向模型
  public static void loadAndForwardModel(final String modelPath) {
    // 调用本地对等体方法，加载并前向模型
    NativePeer.loadAndForwardModel(modelPath);
  }

  private static class NativePeer {
    static {
      // 如果未初始化本地加载器，则使用系统委托进行初始化
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(new SystemDelegate());
      }
      // 加载名为 "pytorch_testapp_jni" 的本地库
      NativeLoader.loadLibrary("pytorch_testapp_jni");
    }

    // 本地方法声明，用于加载并前向模型
    private static native void loadAndForwardModel(final String modelPath);
  }
}
```