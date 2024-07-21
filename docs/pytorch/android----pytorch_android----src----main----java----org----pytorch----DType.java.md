# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\DType.java`

```
/** Codes representing tensor data types. */
public enum DType {
  // NOTE: "jniCode" must be kept in sync with pytorch_jni_common.cpp.
  // NOTE: Never serialize "jniCode", because it can change between releases.

  /** Code for dtype torch.uint8. {@link Tensor#dtype()} */
  UINT8(1),
  /** Code for dtype torch.int8. {@link Tensor#dtype()} */
  INT8(2),
  /** Code for dtype torch.int32. {@link Tensor#dtype()} */
  INT32(3),
  /** Code for dtype torch.float32. {@link Tensor#dtype()} */
  FLOAT32(4),
  /** Code for dtype torch.int64. {@link Tensor#dtype()} */
  INT64(5),
  /** Code for dtype torch.float64. {@link Tensor#dtype()} */
  FLOAT64(6),
  ;

  // 成员变量，存储与 JNI 相关的数据类型代码
  final int jniCode;

  // 枚举类型的构造函数，用于初始化每个枚举常量的 jniCode
  DType(int jniCode) {
    this.jniCode = jniCode;
  }
}
```