# `.\pytorch\android\pytorch_android_torchvision\src\androidTest\java\org\pytorch\torchvision\TorchVisionInstrumentedTests.java`

```py
package org.pytorch.torchvision;

import static org.junit.Assert.assertArrayEquals;

import android.graphics.Bitmap;  // 导入 Android 图片处理类 Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4;  // 导入 Android JUnit4 测试运行器
import org.junit.Test;  // 导入 JUnit 的测试注解
import org.junit.runner.RunWith;  // 导入 JUnit 的运行器注解
import org.pytorch.Tensor;  // 导入 PyTorch 的 Tensor 类

@RunWith(AndroidJUnit4.class)  // 使用 AndroidJUnit4 运行器运行测试
public class TorchVisionInstrumentedTests {

  @Test  // 标注此方法为测试方法
  public void smokeTest() {
    Bitmap bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888);  // 创建一个 ARGB_8888 格式的 Bitmap 对象，尺寸为 320x240
    Tensor tensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,  // 使用预定义的 RGB 均值进行归一化
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);  // 使用预定义的 RGB 标准差进行归一化
    assertArrayEquals(new long[] {1l, 3l, 240l, 320l}, tensor.shape());  // 断言 Tensor 的形状为 [1, 3, 240, 320]
  }
}
```