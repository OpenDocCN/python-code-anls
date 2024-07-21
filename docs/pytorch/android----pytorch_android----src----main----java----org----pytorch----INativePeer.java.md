# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\INativePeer.java`

```
package org.pytorch;

# 定义一个接口 INativePeer，用于表示与本地（native）代码的接口
interface INativePeer {
    # 重置本地代码的状态
    void resetNative();

    # 执行前向推断操作，接收多个输入，并返回一个 IValue 对象
    IValue forward(IValue... inputs);

    # 运行指定方法名的本地方法，接收多个输入，并返回一个 IValue 对象
    IValue runMethod(String methodName, IValue... inputs);
}
```