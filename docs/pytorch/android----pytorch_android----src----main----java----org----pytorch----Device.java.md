# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\Device.java`

```py
package org.pytorch;

public enum Device {
  // 枚举类型定义，表示支持的设备，包括CPU和Vulkan
  CPU(1),        // CPU设备对应的枚举值，JNI代码为1
  VULKAN(2),     // Vulkan设备对应的枚举值，JNI代码为2
  ;

  final int jniCode;   // 定义一个整型变量，用于存储JNI代码

  // 构造方法，初始化枚举常量的JNI代码
  Device(int jniCode) {
    this.jniCode = jniCode;
  }
}
```