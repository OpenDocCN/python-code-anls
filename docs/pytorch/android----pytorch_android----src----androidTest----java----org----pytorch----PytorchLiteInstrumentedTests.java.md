# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\PytorchLiteInstrumentedTests.java`

```
// 导入必要的包，包括 Android 上下文、测试框架和输入输出相关的类
import android.content.Context;
import androidx.test.InstrumentationRegistry;
import androidx.test.runner.AndroidJUnit4;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import org.junit.runner.RunWith;

// 使用 AndroidJUnit4 类作为测试运行器
@RunWith(AndroidJUnit4.class)
// PytorchLiteInstrumentedTests 类继承自 PytorchTestBase 类
public class PytorchLiteInstrumentedTests extends PytorchTestBase {

  // 覆盖基类的方法，加载模型
  @Override
  protected Module loadModel(String path) throws IOException {
    // 调用 LiteModuleLoader 的 load 方法加载模型
    return LiteModuleLoader.load(assetFilePath(path));
  }

  // 根据资产文件名获取文件路径
  private String assetFilePath(String assetName) throws IOException {
    // 获取当前应用的上下文
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    // 在应用的私有文件目录中创建一个文件对象
    File file = new File(appContext.getFilesDir(), assetName);
    // 如果文件存在且文件长度大于0，则返回文件的绝对路径
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    // 如果文件不存在或者文件长度为0，则从应用的 assets 目录中复制文件到私有目录
    try (InputStream is = appContext.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        // 读取输入流并写入输出流，直到输入流结束
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();  // 刷新输出流
      }
      return file.getAbsolutePath();  // 返回文件的绝对路径
    } catch (IOException e) {
      throw e;  // 捕获并重新抛出 IOException
    }
  }
}
```