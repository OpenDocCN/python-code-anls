# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\PytorchHostTests.java`

```py
// 导入所需的 Java 类库
package org.pytorch;

import java.io.IOException;  // 导入 IOException 类
import java.io.InputStream;  // 导入 InputStream 类
import java.nio.file.Files;   // 导入 Files 类
import java.nio.file.Path;    // 导入 Path 类
import java.nio.file.StandardCopyOption;  // 导入 StandardCopyOption 类
import java.util.Objects;     // 导入 Objects 类

// 继承 PytorchTestBase 类，实现 Pytorch 主机测试
public class PytorchHostTests extends PytorchTestBase {

  // 覆盖父类方法，加载模型
  @Override
  protected Module loadModel(String path) throws IOException {
    // 调用 assetFilePath 方法获取模型路径，加载模型并返回
    return Module.load(assetFilePath(path));
  }

  // 私有方法，根据资产名称获取临时文件路径
  private String assetFilePath(String assetName) throws IOException {
    // 创建临时文件，文件名以 "test" 开头，扩展名为 ".pt"
    Path tempFile = Files.createTempFile("test", ".pt");
    // 使用类加载器获取资源流，复制资源到临时文件中
    try (InputStream resource =
        Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("test.pt"))) {
      Files.copy(resource, tempFile, StandardCopyOption.REPLACE_EXISTING);
    }
    // 返回临时文件的绝对路径
    return tempFile.toAbsolutePath().toString();
  }
}
```