# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\PytorchInstrumentedTests.java`

```
package org.pytorch;

import android.content.Context;  // 导入 Android 上下文类
import androidx.test.InstrumentationRegistry;  // 导入 Android 测试工具包
import androidx.test.runner.AndroidJUnit4;  // 导入 Android JUnit4 测试运行器
import java.io.File;  // 导入文件类
import java.io.FileOutputStream;  // 导入文件输出流类
import java.io.IOException;  // 导入输入输出异常类
import java.io.InputStream;  // 导入输入流类
import java.io.OutputStream;  // 导入输出流类
import org.junit.runner.RunWith;  // 导入 JUnit 运行器注解

@RunWith(AndroidJUnit4.class)  // 使用 AndroidJUnit4 作为测试运行器
public class PytorchInstrumentedTests extends PytorchTestBase {

  @Override
  protected Module loadModel(String path) throws IOException {
    return Module.load(assetFilePath(path));  // 调用 assetFilePath 方法加载模型
  }

  private String assetFilePath(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();  // 获取应用的上下文环境
    File file = new File(appContext.getFilesDir(), assetName);  // 在应用文件目录中创建文件对象
    if (file.exists() && file.length() > 0) {  // 如果文件已存在且大小大于0
      return file.getAbsolutePath();  // 返回文件的绝对路径
    }

    try (InputStream is = appContext.getAssets().open(assetName)) {  // 打开 assets 中的输入流
      try (OutputStream os = new FileOutputStream(file)) {  // 创建文件输出流
        byte[] buffer = new byte[4 * 1024];  // 创建缓冲区
        int read;  // 读取的字节数
        while ((read = is.read(buffer)) != -1) {  // 循环读取输入流到缓冲区
          os.write(buffer, 0, read);  // 将缓冲区内容写入文件输出流
        }
        os.flush();  // 刷新输出流
      }
      return file.getAbsolutePath();  // 返回文件的绝对路径
    } catch (IOException e) {  // 捕获输入输出异常
      throw e;  // 抛出异常
    }
  }
}
```