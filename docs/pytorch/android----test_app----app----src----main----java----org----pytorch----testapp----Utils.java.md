# `.\pytorch\android\test_app\app\src\main\java\org\pytorch\testapp\Utils.java`

```py
// 定义了一个名为 org.pytorch.testapp 的包
package org.pytorch.testapp;

// 导入了 Arrays 类，用于数组操作
import java.util.Arrays;

// 定义了一个名为 Utils 的公共类
public class Utils {

  // 定义了一个静态方法 topK，用于返回数组中前 topk 个最大值的索引数组
  public static int[] topK(float[] a, final int topk) {
    // 创建一个长度为 topk 的浮点数数组 values，并用负的最大浮点数填充
    float values[] = new float[topk];
    Arrays.fill(values, -Float.MAX_VALUE);

    // 创建一个长度为 topk 的整数数组 ixs，并用 -1 填充
    int ixs[] = new int[topk];
    Arrays.fill(ixs, -1);

    // 循环遍历输入数组 a
    for (int i = 0; i < a.length; i++) {
      // 内层循环遍历 values 数组
      for (int j = 0; j < topk; j++) {
        // 如果当前元素 a[i] 大于 values[j]
        if (a[i] > values[j]) {
          // 将 values 和 ixs 数组从 j 开始的元素依次后移一位
          for (int k = topk - 1; k >= j + 1; k--) {
            values[k] = values[k - 1];
            ixs[k] = ixs[k - 1];
          }
          // 将 a[i] 插入到 values[j] 的位置，同时记录索引 i 到 ixs[j]
          values[j] = a[i];
          ixs[j] = i;
          // 跳出内层循环
          break;
        }
      }
    }
    // 返回存储 topk 最大值索引的数组 ixs
    return ixs;
  }
}
```