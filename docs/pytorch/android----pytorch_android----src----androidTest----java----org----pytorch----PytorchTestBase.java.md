# `.\pytorch\android\pytorch_android\src\androidTest\java\org\pytorch\PytorchTestBase.java`

```py
// 引入静态方法，用于断言的静态导入
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

// 引入 Java 标准库的异常处理类 IOException
import java.io.IOException;
// 引入用于存储键值对的 HashMap 和 Map 接口
import java.util.HashMap;
import java.util.Map;

// 引入 JUnit 的测试相关类和注解
import org.junit.Test;
import org.junit.Ignore;

// 定义一个抽象类 PytorchTestBase，位于 org.pytorch 包下
public abstract class PytorchTestBase {
  // 定义一个私有静态常量，表示测试模块的名称
  private static final String TEST_MODULE_ASSET_NAME = "android_api_module.ptl";

  // 测试方法：测试模型的正向传播，当输入为 null 时
  @Test
  public void testForwardNull() throws IOException {
    // 载入指定名称的模型，返回一个 Module 实例
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 创建一个输入 IValue，将一个长度为 1 的 ByteBuffer 转换为 Tensor
    final IValue input = IValue.from(Tensor.fromBlob(Tensor.allocateByteBuffer(1), new long[] {1}));
    // 断言输入的 IValue 类型为 Tensor
    assertTrue(input.isTensor());
    // 执行模型的 forward 方法，获取输出的 IValue
    final IValue output = module.forward(input);
    // 断言输出的 IValue 为 null
    assertTrue(output.isNull());
  }

  // 测试方法：测试布尔值的相等性判断
  @Test
  public void testEqBool() throws IOException {
    // 载入指定名称的模型，返回一个 Module 实例
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 遍历布尔值数组
    for (boolean value : new boolean[] {false, true}) {
      // 创建一个输入 IValue，将布尔值转换为 IValue
      final IValue input = IValue.from(value);
      // 断言输入的 IValue 类型为布尔值
      assertTrue(input.isBool());
      // 断言输入的布尔值与 toBool 方法返回的布尔值相等
      assertTrue(value == input.toBool());
      // 执行模型的 eqBool 方法，传入输入的 IValue，获取输出的 IValue
      final IValue output = module.runMethod("eqBool", input);
      // 断言输出的 IValue 类型为布尔值
      assertTrue(output.isBool());
      // 断言输出的布尔值与输入的布尔值相等
      assertTrue(value == output.toBool());
    }
  }

  // 测试方法：测试整数的相等性判断
  @Test
  public void testEqInt() throws IOException {
    // 载入指定名称的模型，返回一个 Module 实例
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 遍历长整型数组
    for (long value : new long[] {Long.MIN_VALUE, -1024, -1, 0, 1, 1024, Long.MAX_VALUE}) {
      // 创建一个输入 IValue，将长整型转换为 IValue
      final IValue input = IValue.from(value);
      // 断言输入的 IValue 类型为长整型
      assertTrue(input.isLong());
      // 断言输入的长整型与 toLong 方法返回的长整型相等
      assertTrue(value == input.toLong());
      // 执行模型的 eqInt 方法，传入输入的 IValue，获取输出的 IValue
      final IValue output = module.runMethod("eqInt", input);
      // 断言输出的 IValue 类型为长整型
      assertTrue(output.isLong());
      // 断言输出的长整型与输入的长整型相等
      assertTrue(value == output.toLong());
    }
  }

  // 测试方法：测试浮点数的相等性判断
  @Test
  public void testEqFloat() throws IOException {
    // 载入指定名称的模型，返回一个 Module 实例
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 定义浮点数数组
    double[] values =
        new double[] {
          -Double.MAX_VALUE,
          Double.MAX_VALUE,
          -Double.MIN_VALUE,
          Double.MIN_VALUE,
          -Math.exp(1.d),
          -Math.sqrt(2.d),
          -3.1415f,
          3.1415f,
          -1,
          0,
          1,
        };
    // 遍历浮点数数组
    for (double value : values) {
      // 创建一个输入 IValue，将浮点数转换为 IValue
      final IValue input = IValue.from(value);
      // 断言输入的 IValue 类型为双精度浮点数
      assertTrue(input.isDouble());
      // 断言输入的浮点数与 toDouble 方法返回的浮点数相等
      assertTrue(value == input.toDouble());
      // 执行模型的 eqFloat 方法，传入输入的 IValue，获取输出的 IValue
      final IValue output = module.runMethod("eqFloat", input);
      // 断言输出的 IValue 类型为双精度浮点数
      assertTrue(output.isDouble());
      // 断言输出的浮点数与输入的浮点数相等
      assertTrue(value == output.toDouble());
    }
  }

  // 测试方法：测试张量的相等性判断
  @Test
  public void testEqTensor() throws IOException {
    // 定义输入张量的形状
    final long[] inputTensorShape = new long[] {1, 3, 224, 224};
    // 计算输入张量的元素总数
    final long numElements = Tensor.numel(inputTensorShape);
    // 创建输入张量的数据数组，并填充数据
    final float[] inputTensorData = new float[(int) numElements];
    for (int i = 0; i < numElements; ++i) {
      inputTensorData[i] = i;
    }
    // 根据数据数组和形状创建张量对象
    final Tensor inputTensor = Tensor.fromBlob(inputTensorData, inputTensorShape);

    // 载入指定名称的模型，返回一个 Module 实例
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);


这段代码还有部分未完成，请问你需要继续还是如何
  @Test
  public void testEqDictIntKeyIntValue() throws IOException {
    // 加载测试模型
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    // 准备输入字典，键为长整型，值为对应的 IValue
    final Map<Long, IValue> inputMap = new HashMap<>();
    inputMap.put(Long.MIN_VALUE, IValue.from(-Long.MIN_VALUE));
    inputMap.put(Long.MAX_VALUE, IValue.from(-Long.MAX_VALUE));
    inputMap.put(0l, IValue.from(0l));
    inputMap.put(1l, IValue.from(-1l));
    inputMap.put(-1l, IValue.from(1l));

    // 创建输入的 IValue 对象，使用长整型键的字典表示
    final IValue input = IValue.dictLongKeyFrom(inputMap);
    // 断言输入的 IValue 对象确实是长整型键的字典
    assertTrue(input.isDictLongKey());

    // 运行模型的特定方法，并获取输出的 IValue 对象
    final IValue output = module.runMethod("eqDictIntKeyIntValue", input);
    // 断言输出的 IValue 对象确实是长整型键的字典
    assertTrue(output.isDictLongKey());

    // 将输出的 IValue 对象转换为长整型键的字典表示
    final Map<Long, IValue> outputMap = output.toDictLongKey();
    // 断言输入字典和输出字典的大小相同
    assertTrue(inputMap.size() == outputMap.size());

    // 遍历输入字典，逐一断言输出字典中每个键对应的值与输入字典中对应的值相同
    for (Map.Entry<Long, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).toLong() == entry.getValue().toLong());
    }
  }

  @Test
  public void testEqDictStrKeyIntValue() throws IOException {
    // 加载测试模型
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    // 准备输入字典，键为字符串，值为对应的 IValue
    final Map<String, IValue> inputMap = new HashMap<>();
    inputMap.put("long_min_value", IValue.from(Long.MIN_VALUE));
    inputMap.put("long_max_value", IValue.from(Long.MAX_VALUE));
    inputMap.put("long_0", IValue.from(0l));
    inputMap.put("long_1", IValue.from(1l));
    inputMap.put("long_-1", IValue.from(-1l));

    // 创建输入的 IValue 对象，使用字符串键的字典表示
    final IValue input = IValue.dictStringKeyFrom(inputMap);
    // 断言输入的 IValue 对象确实是字符串键的字典
    assertTrue(input.isDictStringKey());

    // 运行模型的特定方法，并获取输出的 IValue 对象
    final IValue output = module.runMethod("eqDictStrKeyIntValue", input);
    // 断言输出的 IValue 对象确实是字符串键的字典
    assertTrue(output.isDictStringKey());

    // 将输出的 IValue 对象转换为字符串键的字典表示
    final Map<String, IValue> outputMap = output.toDictStringKey();
    // 断言输入字典和输出字典的大小相同
    assertTrue(inputMap.size() == outputMap.size());

    // 遍历输入字典，逐一断言输出字典中每个键对应的值与输入字典中对应的值相同
    for (Map.Entry<String, IValue> entry : inputMap.entrySet()) {
      assertTrue(outputMap.get(entry.getKey()).toLong() == entry.getValue().toLong());
    }
  }
  // 使用一个包含三个元素的整数数组进行迭代：0、1、128
  for (int n : new int[] {0, 1, 128}) {
    // 创建一个包含 n 个元素的长整型数组 a
    long[] a = new long[n];
    // 初始化变量 sum 用于累加数组 a 的元素和
    long sum = 0;
    // 遍历数组 a
    for (int i = 0; i < n; i++) {
      // 将数组 a 的第 i 个元素赋值为 i
      a[i] = i;
      // 将当前元素 i 累加到 sum 中
      sum += a[i];
    }
    // 将长整型数组 a 转换为 IValue 类型的列表 input
    final IValue input = IValue.listFrom(a);
    // 断言 input 是一个长整型列表
    assertTrue(input.isLongList());

    // 使用模块运行方法 "listIntSumReturnTuple"，传入参数 input
    final IValue output = module.runMethod("listIntSumReturnTuple", input);

    // 断言 output 是一个元组
    assertTrue(output.isTuple());
    // 断言 output 元组的长度为 2
    assertTrue(2 == output.toTuple().length);

    // 获取 output 元组的第一个元素
    IValue output0 = output.toTuple()[0];
    // 获取 output 元组的第二个元素
    IValue output1 = output.toTuple()[1];

    // 断言数组 a 与 output0 相等
    assertArrayEquals(a, output0.toLongList());
    // 断言 sum 与 output1 相等
    assertTrue(sum == output1.toLong());
  }
}

@Test
public void testOptionalIntIsNone() throws IOException {
  // 加载指定名称的模块
  final Module module = loadModel(TEST_MODULE_ASSET_NAME);

  // 断言模块运行方法 "optionalIntIsNone"，传入参数 1L 的结果为 false
  assertFalse(module.runMethod("optionalIntIsNone", IValue.from(1L)).toBool());
  // 断言模块运行方法 "optionalIntIsNone"，传入可选空值的结果为 true
  assertTrue(module.runMethod("optionalIntIsNone", IValue.optionalNull()).toBool());
}

@Test
public void testIntEq0None() throws IOException {
  // 加载指定名称的模块
  final Module module = loadModel(TEST_MODULE_ASSET_NAME);

  // 断言模块运行方法 "intEq0None"，传入参数 0L 的结果为 null
  assertTrue(module.runMethod("intEq0None", IValue.from(0L)).isNull());
  // 断言模块运行方法 "intEq0None"，传入参数 1L 的结果为 1L
  assertTrue(module.runMethod("intEq0None", IValue.from(1L)).toLong() == 1L);
}

@Test(expected = IllegalArgumentException.class)
public void testRunUndefinedMethod() throws IOException {
  // 加载指定名称的模块
  final Module module = loadModel(TEST_MODULE_ASSET_NAME);
  // 断言模块运行方法 "test_undefined_method_throws_exception" 会抛出 IllegalArgumentException 异常
  module.runMethod("test_undefined_method_throws_exception");
}

@Test
public void testTensorMethods() {
  // 定义张量的形状
  long[] shape = new long[] {1, 3, 224, 224};
  // 计算张量的元素总数
  final int numel = (int) Tensor.numel(shape);
  // 创建一个整数数组 ints，长度为 numel
  int[] ints = new int[numel];
  // 创建一个浮点数数组 floats，长度为 numel
  float[] floats = new float[numel];

  // 创建一个字节数组 bytes，长度为 numel
  byte[] bytes = new byte[numel];
  // 循环初始化 bytes、ints 和 floats 数组
  for (int i = 0; i < numel; i++) {
    // 计算 bytes 数组元素的值
    bytes[i] = (byte) ((i % 255) - 128);
    // 将 i 赋值给 ints 数组的第 i 个元素
    ints[i] = i;
    // 计算 floats 数组的第 i 个元素
    floats[i] = i / 1000.f;
  }

  // 根据 bytes 和 shape 创建张量 tensorBytes
  Tensor tensorBytes = Tensor.fromBlob(bytes, shape);
  // 断言 tensorBytes 的数据类型为 INT8
  assertTrue(tensorBytes.dtype() == DType.INT8);
  // 断言 tensorBytes 的数据与 bytes 数组相等
  assertArrayEquals(bytes, tensorBytes.getDataAsByteArray());

  // 根据 ints 和 shape 创建张量 tensorInts
  Tensor tensorInts = Tensor.fromBlob(ints, shape);
  // 断言 tensorInts 的数据类型为 INT32
  assertTrue(tensorInts.dtype() == DType.INT32);
  // 断言 tensorInts 的数据与 ints 数组相等
  assertArrayEquals(ints, tensorInts.getDataAsIntArray());

  // 根据 floats 和 shape 创建张量 tensorFloats
  Tensor tensorFloats = Tensor.fromBlob(floats, shape);
  // 断言 tensorFloats 的数据类型为 FLOAT32
  assertTrue(tensorFloats.dtype() == DType.FLOAT32);
  // 获取 tensorFloats 的数据，并与 floats 数组逐个元素比较
  float[] floatsOut = tensorFloats.getDataAsFloatArray();
  assertTrue(floatsOut.length == numel);
  for (int i = 0; i < numel; i++) {
    assertTrue(floats[i] == floatsOut[i]);
  }
}

@Test(expected = IllegalStateException.class)
public void testTensorIllegalStateOnWrongType() {
  // 定义张量的形状
  long[] shape = new long[] {1, 3, 224, 224};
  // 计算张量的元素总数
  final int numel = (int) Tensor.numel(shape);
  // 创建一个浮点数数组 floats，长度为 numel
  float[] floats = new float[numel];
  // 根据 floats 和 shape 创建张量 tensorFloats
  Tensor tensorFloats = Tensor.fromBlob(floats, shape);
  // 断言 tensorFloats 的数据类型为 FLOAT32
  assertTrue(tensorFloats.dtype() == DType.FLOAT32);
  // 尝试将 FLOAT32 类型的张量转换为字节数组，预期会抛出 IllegalStateException 异常
  tensorFloats.getDataAsByteArray();
}

@Test
public void testEqString() throws IOException {
  // 加载指定名称的模块
  final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 定义一个包含多个字符串的数组，用于测试不同类型的字符串输入
    String[] values =
        new String[] {
          "smoketest",  // 普通拉丁字符测试
          "проверка не латинских символов", // 检查非拉丁字符
          "#@$!@#)($*!@#$)(!@*#$"  // 特殊字符测试
        };
    // 遍历字符串数组，对每个字符串进行以下操作
    for (String value : values) {
      // 将字符串转换为特定类型的值对象
      final IValue input = IValue.from(value);
      // 断言该值对象是字符串类型
      assertTrue(input.isString());
      // 确保值对象转换为字符串后与原始字符串相同
      assertTrue(value.equals(input.toStr()));
      // 运行模块中的方法，并传入当前的值对象作为参数，获取输出值对象
      final IValue output = module.runMethod("eqStr", input);
      // 断言输出值对象仍然是字符串类型
      assertTrue(output.isString());
      // 确保输出值对象转换为字符串后与原始字符串相同
      assertTrue(value.equals(output.toStr()));
    }
  }

  @Test
  public void testStr3Concat() throws IOException {
    // 加载测试模块
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 定义一个包含多个字符串的数组，用于测试字符串的三次串联操作
    String[] values =
        new String[] {
          "smoketest",  // 普通拉丁字符测试
          "проверка не латинских символов", // 检查非拉丁字符
          "#@$!@#)($*!@#$)(!@*#$"  // 特殊字符测试
        };
    // 遍历字符串数组，对每个字符串进行以下操作
    for (String value : values) {
      // 将字符串转换为特定类型的值对象
      final IValue input = IValue.from(value);
      // 断言该值对象是字符串类型
      assertTrue(input.isString());
      // 确保值对象转换为字符串后与原始字符串相同
      assertTrue(value.equals(input.toStr()));
      // 运行模块中的方法，并传入当前的值对象作为参数，获取输出值对象
      final IValue output = module.runMethod("str3Concat", input);
      // 断言输出值对象仍然是字符串类型
      assertTrue(output.isString());
      // 构建预期的输出字符串，三次串联当前字符串
      String expectedOutput =
          new StringBuilder().append(value).append(value).append(value).toString();
      // 确保输出值对象转换为字符串后与预期的三次串联字符串相同
      assertTrue(expectedOutput.equals(output.toStr()));
    }
  }

  @Test
  public void testEmptyShape() throws IOException {
    // 加载测试模块
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 定义一个数字作为输入的形状
    final long someNumber = 43;
    // 将数字包装为张量的值对象，并指定空的形状
    final IValue input = IValue.from(Tensor.fromBlob(new long[] {someNumber}, new long[] {}));
    // 运行模块中的方法，并传入当前的值对象作为参数，获取输出值对象
    final IValue output = module.runMethod("newEmptyShapeWithItem", input);
    // 断言输出值对象是张量类型
    assertTrue(output.isTensor());
    // 将输出值对象转换为张量，并断言其形状为空数组
    Tensor value = output.toTensor();
    assertArrayEquals(new long[] {}, value.shape());
    // 断言张量的数据与输入的数字相同
    assertArrayEquals(new long[] {someNumber}, value.getDataAsLongArray());
  }

  @Test
  public void testAliasWithOffset() throws IOException {
    // 加载测试模块
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 运行模块中的方法，获取输出值对象
    final IValue output = module.runMethod("testAliasWithOffset");
    // 断言输出值对象是张量列表类型
    assertTrue(output.isTensorList());
    // 将输出值对象转换为张量列表，并断言第一个张量的数据与预期值相同
    Tensor[] tensors = output.toTensorList();
    assertEquals(100, tensors[0].getDataAsLongArray()[0]);
    assertEquals(200, tensors[1].getDataAsLongArray()[0]);
  }

  @Test
  public void testNonContiguous() throws IOException {
    // 加载测试模块
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 运行模块中的方法，获取输出值对象
    final IValue output = module.runMethod("testNonContiguous");
    // 断言输出值对象是张量类型
    assertTrue(output.isTensor());
    // 将输出值对象转换为张量，并断言其形状为 [2]
    Tensor value = output.toTensor();
    assertArrayEquals(new long[] {2}, value.shape());
    // 断言张量的数据与预期值相同
    assertArrayEquals(new long[] {100, 300}, value.getDataAsLongArray());
  }

  @Test
  public void testChannelsLast() throws IOException {
    // 定义输入张量的形状和数据
    long[] inputShape = new long[] {1, 3, 2, 2};
    long[] data = new long[] {1, 11, 101, 2, 12, 102, 3, 13, 103, 4, 14, 104};
    // 创建张量对象，指定数据、形状和内存布局为 CHANNELS_LAST
    Tensor inputNHWC = Tensor.fromBlob(data, inputShape, MemoryFormat.CHANNELS_LAST);
    // 加载测试模块
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 运行模块的 "contiguous" 方法，将输入张量从 NHWC 格式转换为 NCHW 格式
    final IValue outputNCHW = module.runMethod("contiguous", IValue.from(inputNHWC));
    // 断言输出的张量格式为 CONTIGUOUS，形状为 [1, 3, 2, 2]，数据为给定的数组
    assertIValueTensor(
        outputNCHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2},
        new long[] {1, 2, 3, 4, 11, 12, 13, 14, 101, 102, 103, 104});

    // 运行模块的 "contiguousChannelsLast" 方法，将输入张量从 NHWC 格式转换为 CHANNELS_LAST 格式
    final IValue outputNHWC = module.runMethod("contiguousChannelsLast", IValue.from(inputNHWC));
    // 断言输出的张量格式为 CHANNELS_LAST，形状与输入形状相同，数据为给定的输入数据数组
    assertIValueTensor(outputNHWC, MemoryFormat.CHANNELS_LAST, inputShape, data);
  }

  @Test
  public void testChannelsLast3d() throws IOException {
    // 定义形状和数据数组，创建 CHANNELS_LAST_3D 格式的输入张量
    long[] shape = new long[] {1, 2, 2, 2, 2};
    long[] dataNCHWD = new long[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    long[] dataNHWDC = new long[] {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16};

    // 根据数据创建张量，指定 MEMORY_FORMAT 为 CHANNELS_LAST_3D
    Tensor inputNHWDC = Tensor.fromBlob(dataNHWDC, shape, MemoryFormat.CHANNELS_LAST_3D);
    // 加载模型
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);
    // 运行模块的 "contiguous" 方法，将输入张量从 NHWDC 格式转换为 NCHWD 格式
    final IValue outputNCHWD = module.runMethod("contiguous", IValue.from(inputNHWDC));
    // 断言输出的张量格式为 CONTIGUOUS，形状为给定形状，数据为给定的 dataNCHWD 数组
    assertIValueTensor(outputNCHWD, MemoryFormat.CONTIGUOUS, shape, dataNCHWD);

    // 创建 CONTIGUOUS 格式的输入张量
    Tensor inputNCHWD = Tensor.fromBlob(dataNCHWD, shape, MemoryFormat.CONTIGUOUS);
    // 运行模块的 "contiguousChannelsLast3d" 方法，将输入张量从 NCHWD 格式转换为 CHANNELS_LAST_3D 格式
    final IValue outputNHWDC =
        module.runMethod("contiguousChannelsLast3d", IValue.from(inputNCHWD));
    // 断言输出的张量格式为 CHANNELS_LAST_3D，形状为给定形状，数据为给定的 dataNHWDC 数组
    assertIValueTensor(outputNHWDC, MemoryFormat.CHANNELS_LAST_3D, shape, dataNHWDC);
  }

  @Test
  public void testChannelsLastConv2d() throws IOException {
    // 定义输入形状和数据数组，创建 CONTIGUOUS 格式的输入张量
    long[] inputShape = new long[] {1, 3, 2, 2};
    long[] dataNCHW = new long[] {
      111, 112,
      121, 122,

      211, 212,
      221, 222,

      311, 312,
      321, 322};
    Tensor inputNCHW = Tensor.fromBlob(dataNCHW, inputShape, MemoryFormat.CONTIGUOUS);
    // 定义输入数据数组，创建 CHANNELS_LAST 格式的输入张量
    long[] dataNHWC = new long[] {
      111, 211, 311,       112, 212, 312,

      121, 221, 321,       122, 222, 322};
    Tensor inputNHWC = Tensor.fromBlob(dataNHWC, inputShape, MemoryFormat.CHANNELS_LAST);
    // 定义权重形状和数据数组，创建 CONTIGUOUS 格式的权重张量
    long[] weightShape = new long[] {3, 3, 1, 1};
    long[] dataWeightOIHW = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1};
    Tensor wNCHW = Tensor.fromBlob(dataWeightOIHW, weightShape, MemoryFormat.CONTIGUOUS);
    // 定义权重数据数组，创建 CHANNELS_LAST 格式的权重张量
    long[] dataWeightOHWI = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1};
    Tensor wNHWC = Tensor.fromBlob(dataWeightOHWI, weightShape, MemoryFormat.CHANNELS_LAST);

    // 加载模型
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    // 运行模块的 "conv2d" 方法，进行卷积操作，输出 CONTIGUOUS 格式的张量
    final IValue outputNCHW =
        module.runMethod("conv2d", IValue.from(inputNCHW), IValue.from(wNCHW), IValue.from(false));
    // 断言输出的张量格式为 CONTIGUOUS，形状为 [1, 3, 2, 2]，数据为给定的数组
    assertIValueTensor(
        outputNCHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2},
        new long[] {
          2*111, 2*112,
          2*121, 2*122,

          211, 212,
          221, 222,

          -311, -312,
          -321, -322});

    // 运行模块的 "conv2d" 方法，进行卷积操作，输出 CHANNELS_LAST 格式的张量
    final IValue outputNHWC =
        module.runMethod("conv2d", IValue.from(inputNHWC), IValue.from(wNHWC), IValue.from(true));
  @Test
  // 测试 Channels Last 格式的 3D 卷积
  public void testChannelsLastConv3d() throws IOException {
    // 定义输入张量的形状为 [1, 3, 2, 2, 2]
    long[] inputShape = new long[] {1, 3, 2, 2, 2};
    // 输入数据按照 NCDHW 格式排列
    long[] dataNCDHW = new long[] {
      1111, 1112,
      1121, 1122,
      1211, 1212,
      1221, 1222,

      2111, 2112,
      2121, 2122,
      2211, 2212,
      2221, 2222,

      3111, 3112,
      3121, 3122,
      3211, 3212,
      3221, 3222};
    // 创建 NCDHW 格式的输入张量
    Tensor inputNCDHW = Tensor.fromBlob(dataNCDHW, inputShape, MemoryFormat.CONTIGUOUS);
    // 输入数据按照 NDHWC 格式排列
    long[] dataNDHWC = new long[] {
      1111, 2111, 3111,
      1112, 2112, 3112,

      1121, 2121, 3121,
      1122, 2122, 3122,

      1211, 2211, 3211,
      1212, 2212, 3212,

      1221, 2221, 3221,
      1222, 2222, 3222};
    // 创建 NDHWC 格式的输入张量
    Tensor inputNDHWC = Tensor.fromBlob(dataNDHWC, inputShape, MemoryFormat.CHANNELS_LAST_3D);

    // 定义卷积核的形状为 [3, 3, 1, 1, 1]
    long[] weightShape = new long[] {3, 3, 1, 1, 1};
    // 定义 NCDHW 格式的卷积核数据
    long[] dataWeightOIDHW = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1,
    };
    // 创建 NCDHW 格式的卷积核张量
    Tensor wNCDHW = Tensor.fromBlob(dataWeightOIDHW, weightShape, MemoryFormat.CONTIGUOUS);
    // 定义 NDHWC 格式的卷积核数据
    long[] dataWeightODHWI = new long[] {
      2, 0, 0,
      0, 1, 0,
      0, 0, -1,
    };
    // 创建 NDHWC 格式的卷积核张量
    Tensor wNDHWC = Tensor.fromBlob(dataWeightODHWI, weightShape, MemoryFormat.CHANNELS_LAST_3D);

    // 加载模型
    final Module module = loadModel(TEST_MODULE_ASSET_NAME);

    // 对输入张量 inputNCDHW 进行 3D 卷积运算
    final IValue outputNCDHW =
        module.runMethod("conv3d", IValue.from(inputNCDHW), IValue.from(wNCDHW), IValue.from(false));
    // 断言输出张量的内存格式为 CONTIGUOUS，期望形状为 [1, 3, 2, 2, 2]
    assertIValueTensor(
        outputNCDHW,
        MemoryFormat.CONTIGUOUS,
        new long[] {1, 3, 2, 2, 2},
        new long[] {
          2*1111, 2*1112,     2*1121, 2*1122,
          2*1211, 2*1212,     2*1221, 2*1222,

          2111, 2112,     2121, 2122,
          2211, 2212,     2221, 2222,

          -3111, -3112,     -3121, -3122,
          -3211, -3212,     -3221, -3222});

    // 对输入张量 inputNDHWC 进行 3D 卷积运算
    final IValue outputNDHWC =
        module.runMethod("conv3d", IValue.from(inputNDHWC), IValue.from(wNDHWC), IValue.from(true));
    // 断言输出张量的内存格式为 CHANNELS_LAST_3D，期望形状为 [1, 3, 2, 2, 2]
    assertIValueTensor(
        outputNDHWC,
        MemoryFormat.CHANNELS_LAST_3D,
        new long[] {1, 3, 2, 2, 2},
        new long[] {
          2*1111, 2111, -3111,      2*1112, 2112, -3112,
          2*1121, 2121, -3121,      2*1122, 2122, -3122,

          2*1211, 2211, -3211,      2*1212, 2212, -3212,
          2*1221, 2221, -3221,      2*1222, 2222, -3222});
  }
    @Test
    public void testPointwiseOps() throws IOException {
      // 执行名为 "pointwise_ops" 的模型测试
      runModel("pointwise_ops");
    }
    
    @Test
    public void testReductionOps() throws IOException {
      // 执行名为 "reduction_ops" 的模型测试
      runModel("reduction_ops");
    }
    
    @Test
    public void testComparisonOps() throws IOException {
      // 执行名为 "comparison_ops" 的模型测试
      runModel("comparison_ops");
    }
    
    @Test
    public void testOtherMathOps() throws IOException {
      // 执行名为 "other_math_ops" 的模型测试
      runModel("other_math_ops");
    }
    
    @Test
    @Ignore
    public void testSpectralOps() throws IOException {
      // NB: This model fails without lite interpreter.  The error is as follows:
      // RuntimeError: stft requires the return_complex parameter be given for real inputs
      // 执行名为 "spectral_ops" 的模型测试，注意需要 Lite 解释器才能成功运行，否则会失败并报错
      runModel("spectral_ops");
    }
    
    @Test
    public void testBlasLapackOps() throws IOException {
      // 执行名为 "blas_lapack_ops" 的模型测试
      runModel("blas_lapack_ops");
    }
    
    @Test
    public void testSamplingOps() throws IOException {
      // 执行名为 "sampling_ops" 的模型测试
      runModel("sampling_ops");
    }
    
    @Test
    public void testTensorOps() throws IOException {
      // 执行名为 "tensor_general_ops" 的模型测试
      runModel("tensor_general_ops");
    }
    
    @Test
    public void testTensorCreationOps() throws IOException {
      // 执行名为 "tensor_creation_ops" 的模型测试
      runModel("tensor_creation_ops");
    }
    
    @Test
    public void testTensorIndexingOps() throws IOException {
      // 执行名为 "tensor_indexing_ops" 的模型测试
      runModel("tensor_indexing_ops");
    }
    
    @Test
    public void testTensorTypingOps() throws IOException {
      // 执行名为 "tensor_typing_ops" 的模型测试
      runModel("tensor_typing_ops");
    }
    
    @Test
    public void testTensorViewOps() throws IOException {
      // 执行名为 "tensor_view_ops" 的模型测试
      runModel("tensor_view_ops");
    }
    
    @Test
    public void testConvolutionOps() throws IOException {
      // 执行名为 "convolution_ops" 的模型测试
      runModel("convolution_ops");
    }
    
    @Test
    public void testPoolingOps() throws IOException {
      // 执行名为 "pooling_ops" 的模型测试
      runModel("pooling_ops");
    }
    
    @Test
    public void testPaddingOps() throws IOException {
      // 执行名为 "padding_ops" 的模型测试
      runModel("padding_ops");
    }
    
    @Test
    public void testActivationOps() throws IOException {
      // 执行名为 "activation_ops" 的模型测试
      runModel("activation_ops");
    }
    
    @Test
    public void testNormalizationOps() throws IOException {
      // 执行名为 "normalization_ops" 的模型测试
      runModel("normalization_ops");
    }
    
    @Test
    public void testRecurrentOps() throws IOException {
      // 执行名为 "recurrent_ops" 的模型测试
      runModel("recurrent_ops");
    }
    
    @Test
    public void testTransformerOps() throws IOException {
      // 执行名为 "transformer_ops" 的模型测试
      runModel("transformer_ops");
    }
    
    @Test
    public void testLinearOps() throws IOException {
      // 执行名为 "linear_ops" 的模型测试
      runModel("linear_ops");
    }
    
    @Test
    public void testDropoutOps() throws IOException {
      // 执行名为 "dropout_ops" 的模型测试
      runModel("dropout_ops");
    }
    
    @Test
    public void testSparseOps() throws IOException {
      // 执行名为 "sparse_ops" 的模型测试
      runModel("sparse_ops");
    }
    
    @Test
    public void testDistanceFunctionOps() throws IOException {
      // 执行名为 "distance_function_ops" 的模型测试
      runModel("distance_function_ops");
    }
    
    @Test
    public void testLossFunctionOps() throws IOException {
      // 执行名为 "loss_function_ops" 的模型测试
      runModel("loss_function_ops");
    }
    
    @Test
    public void testVisionFunctionOps() throws IOException {
      // 执行名为 "vision_function_ops" 的模型测试
      runModel("vision_function_ops");
    }
    
    @Test
    public void testShuffleOps() throws IOException {
      // 执行名为 "shuffle_ops" 的模型测试
      runModel("shuffle_ops");
    }
    
    @Test
    public void testNNUtilsOps() throws IOException {
  // 运行指定名称的模型，名称由参数 `name` 提供
  void runModel(final String name) throws IOException {
    // 载入指定名称的模型文件，并创建模块对象
    final Module storage_module = loadModel(name + ".ptl");
    // 执行模型的前向传播计算
    storage_module.forward();

    // TODO: 在临时脚本准备好后启用这段代码
    // final Module on_the_fly_module = loadModel(name + "_temp.ptl");
    // on_the_fly_module.forward();
    
    // 断言总是为真，用于确认测试执行的完整性
    assertTrue(true);
  }

  // 断言 IValue 是一个张量，验证其内存格式、形状和数据是否与期望值匹配
  static void assertIValueTensor(
      final IValue ivalue,
      final MemoryFormat memoryFormat,
      final long[] expectedShape,
      final long[] expectedData) {
    // 断言 IValue 是一个张量对象
    assertTrue(ivalue.isTensor());
    // 获取 IValue 中的张量对象
    Tensor t = ivalue.toTensor();
    // 验证张量的内存格式是否与期望相符
    assertEquals(memoryFormat, t.memoryFormat());
    // 验证张量的形状是否与期望相符
    assertArrayEquals(expectedShape, t.shape());
    // 验证张量的数据内容是否与期望相符
    assertArrayEquals(expectedData, t.getDataAsLongArray());
  }

  // 测试函数：运行指定名称的模型，名称由参数 `name` 提供
  @Test
  public void testQuantOps() throws IOException {
    runModel("general_quant_ops");
  }

  // 测试函数：运行动态量化操作的模型
  @Test
  public void testDynamicQuantOps() throws IOException {
    runModel("dynamic_quant_ops");
  }

  // 测试函数：运行静态量化操作的模型
  @Test
  public void testStaticQuantOps() throws IOException {
    runModel("static_quant_ops");
  }

  // 测试函数：运行融合量化操作的模型
  @Test
  public void testFusedQuantOps() throws IOException {
    runModel("fused_quant_ops");
  }

  // 测试函数：运行 TorchScript 内置量化操作的模型
  @Test
  public void testTorchScriptBuiltinQuantOps() throws IOException {
    runModel("torchscript_builtin_ops");
  }

  // 测试函数：运行 TorchScript 集合量化操作的模型
  @Test
  public void testTorchScriptCollectionQuantOps() throws IOException {
    runModel("torchscript_collection_ops");
  }

  // 抽象方法：载入指定名称的模型并返回模块对象
  protected abstract Module loadModel(String assetName) throws IOException;
}


注释：


# 这行代码关闭了一个代码块，通常用于结束一个函数或条件语句的定义。
```