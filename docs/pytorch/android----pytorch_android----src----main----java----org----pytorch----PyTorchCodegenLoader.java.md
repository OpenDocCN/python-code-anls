# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\PyTorchCodegenLoader.java`

```
# 导入必要的包
package org.pytorch;

# 导入 NativeLoader 类，该类用于加载本地库
import com.facebook.soloader.nativeloader.NativeLoader;

# 定义 PyTorchCodegenLoader 类
public class PyTorchCodegenLoader {

  # 静态方法：加载本地库
  public static void loadNativeLibs() {
    try {
      # 使用 NativeLoader 加载名为 "torch-code-gen" 的本地库
      NativeLoader.loadLibrary("torch-code-gen");
    } catch (Throwable t) {
      # 捕获可能出现的任何异常，注释说明加载代码生成库是最佳尝试，因为它仅用于基于查询的构建。
      // Loading the codegen lib is best-effort since it's only there for query based builds.
    }
  }

  # 私有构造函数，防止类实例化
  private PyTorchCodegenLoader() {}
}
```