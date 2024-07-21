# `.\pytorch\android\pytorch_android\src\main\java\org\pytorch\Tensor.java`

```
  // 导入必要的类和接口
  package org.pytorch;

  import com.facebook.jni.HybridData;
  import com.facebook.jni.annotations.DoNotStrip;
  import java.nio.Buffer;
  import java.nio.ByteBuffer;
  import java.nio.ByteOrder;
  import java.nio.DoubleBuffer;
  import java.nio.FloatBuffer;
  import java.nio.IntBuffer;
  import java.nio.LongBuffer;
  import java.util.Arrays;
  import java.util.Locale;

  /**
   * 表示一个张量（Tensor）。其行为类似于 PyTorch 的张量对象。
   *
   * <p>大多数张量将通过 {@code Tensor.fromBlob(data, shape)} 构造，其中 {@code data} 可以是数组或直接的 {@link Buffer}（适当的子类）。
   * 提供了帮助方法以正确分配缓冲区。
   *
   * <p>要访问张量数据，请参阅 {@link #dtype()}、{@link #shape()} 和各种 {@code getDataAs*} 方法。
   *
   * <p>当使用数组作为 {@code data} 构造 {@code Tensor} 对象时，未指定是否复制此数据或保留其引用，因此建议在构造后不要修改它。
   * 传递 {@link Buffer} 作为 {@code data} 不会复制，因此可以在 {@link Module} 调用之间修改以避免重新分配。
   * 从 {@code Tensor} 对象检索的数据可能被复制，也可能是对 {@code Tensor} 内部数据缓冲区的引用。
   * {@code shape} 总是被复制。
   */
  public abstract class Tensor {
    // 数据缓冲区不能为空的错误消息
    private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
    // 数据数组不能为空的错误消息
    private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
    // 形状不能为空的错误消息
    private static final String ERROR_MSG_SHAPE_NOT_NULL = "Shape must be not null";
    // 形状元素必须为非负数的错误消息
    private static final String ERROR_MSG_SHAPE_NON_NEGATIVE = "Shape elements must be non negative";
    // 数据缓冲区必须具有本地字节顺序的错误消息
    private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
        "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
    // 数据缓冲区必须是直接缓冲区的错误消息
    private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
        "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

    // 不要剥离注释的最终形式
    @DoNotStrip final long[] shape;
    // 存储格式的内存格式
    final MemoryFormat memoryFormat;

    // 整数大小的字节数
    private static final int INT_SIZE_BYTES = 4;
    // 浮点数大小的字节数
    private static final int FLOAT_SIZE_BYTES = 4;
    // 长整型大小的字节数
    private static final int LONG_SIZE_BYTES = 8;
    // 双精度浮点数大小的字节数
    private static final int DOUBLE_SIZE_BYTES = 8;

    /**
     * 分配一个新的具有本地字节顺序的直接 {@link java.nio.ByteBuffer}，其指定容量可用于 {@link Tensor#fromBlob(ByteBuffer, long[])}、
     * {@link Tensor#fromBlobUnsigned(ByteBuffer, long[])}。
     *
     * @param numElements 结果缓冲区的容量（元素数量）。
     */
    public static ByteBuffer allocateByteBuffer(int numElements) {
    // 分配一个新的直接 ByteBuffer，其容量为 numElements，使用本地字节顺序
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  /**
   * 使用指定容量分配一个新的直接 {@link java.nio.IntBuffer}，该缓冲区可以在 {@link Tensor#fromBlob(IntBuffer, long[])} 中使用。
   *
   * @param numElements 结果缓冲区的容量（元素数量）。
   */
  public static IntBuffer allocateIntBuffer(int numElements) {
    // 分配一个新的直接 ByteBuffer，其容量为 numElements * INT_SIZE_BYTES，使用本地字节顺序，并转换为 IntBuffer
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /**
   * 使用指定容量分配一个新的直接 {@link java.nio.FloatBuffer}，该缓冲区可以在 {@link Tensor#fromBlob(FloatBuffer, long[])} 中使用。
   *
   * @param numElements 结果缓冲区的容量（元素数量）。
   */
  public static FloatBuffer allocateFloatBuffer(int numElements) {
    // 分配一个新的直接 ByteBuffer，其容量为 numElements * FLOAT_SIZE_BYTES，使用本地字节顺序，并转换为 FloatBuffer
    return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  /**
   * 使用指定容量分配一个新的直接 {@link java.nio.LongBuffer}，该缓冲区可以在 {@link Tensor#fromBlob(LongBuffer, long[])} 中使用。
   *
   * @param numElements 结果缓冲区的容量（元素数量）。
   */
  public static LongBuffer allocateLongBuffer(int numElements) {
    // 分配一个新的直接 ByteBuffer，其容量为 numElements * LONG_SIZE_BYTES，使用本地字节顺序，并转换为 LongBuffer
    return ByteBuffer.allocateDirect(numElements * LONG_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer();
  }

  /**
   * 使用指定容量分配一个新的直接 {@link java.nio.DoubleBuffer}，该缓冲区可以在 {@link Tensor#fromBlob(DoubleBuffer, long[])} 中使用。
   *
   * @param numElements 结果缓冲区的容量（元素数量）。
   */
  public static DoubleBuffer allocateDoubleBuffer(int numElements) {
    // 分配一个新的直接 ByteBuffer，其容量为 numElements * DOUBLE_SIZE_BYTES，使用本地字节顺序，并转换为 DoubleBuffer
    return ByteBuffer.allocateDirect(numElements * DOUBLE_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asDoubleBuffer();
  }

  /**
   * 使用指定的数据和形状数组创建一个 dtype 为 torch.uint8 的新 Tensor 实例。
   *
   * @param data Tensor 的元素数据
   * @param shape Tensor 的形状
   * @param memoryFormat 内存格式
   * @throws IllegalArgumentException 如果数据数组或形状数组为 null
   * @throws IllegalArgumentException 如果形状不合法
   * @throws IllegalArgumentException 如果数据数组的长度与形状要求的容量不一致
   */
  public static Tensor fromBlobUnsigned(byte[] data, long[] shape, MemoryFormat memoryFormat) {
    // 检查数据和形状数组不为 null
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状合法性
    checkShape(shape);
    // 检查数据长度与形状要求的容量一致性
    checkShapeAndDataCapacityConsistency(data.length, shape);
    // 分配一个新的 ByteBuffer，其容量为 shape 元素的数量，并放入数据
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    // 返回一个新的 Tensor_uint8 实例
    return new Tensor_uint8(byteBuffer, shape, memoryFormat);
  }

  // 另一个 fromBlobUnsigned 方法可能被省略，这里不需要注释
    return fromBlobUnsigned(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 使用无符号数据创建一个新的 Tensor 实例，数据类型为 torch.uint8，指定形状和内存格式。
   *
   * @param data Tensor 元素的字节数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(byte[] data, long[] shape, MemoryFormat memoryFormat) {
    // 检查数据数组不为空
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    // 检查形状数组不为空
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状的有效性
    checkShape(shape);
    // 检查数据容量与形状一致性
    checkShapeAndDataCapacityConsistency(data.length, shape);
    // 分配一个字节缓冲区，并将数据放入其中
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    // 使用分配的字节缓冲区、指定的形状和内存格式创建新的 Tensor_int8 实例
    return new Tensor_int8(byteBuffer, shape, memoryFormat);
  }

  /**
   * 使用默认的内存格式（CONTIGUOUS）创建一个新的 Tensor 实例，数据类型为 torch.int8，指定形状和数据数组。
   *
   * @param data Tensor 元素的字节数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(byte[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 创建一个新的 Tensor 实例，数据类型为 torch.int32，指定形状和数据数组。
   *
   * @param data Tensor 元素的整数数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(int[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final IntBuffer intBuffer = allocateIntBuffer((int) numel(shape));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, shape, memoryFormat);
  }

  /**
   * 使用默认的内存格式（CONTIGUOUS）创建一个新的 Tensor 实例，数据类型为 torch.int32，指定形状和数据数组。
   *
   * @param data Tensor 元素的整数数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(int[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 创建一个新的 Tensor 实例，数据类型为 torch.float32，指定形状和数据数组。
   *
   * @param data Tensor 元素的浮点数数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(float[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final FloatBuffer floatBuffer = allocateFloatBuffer((int) numel(shape));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, shape, memoryFormat);
  }

  /**
   * 使用默认的内存格式（CONTIGUOUS）创建一个新的 Tensor 实例，数据类型为 torch.float32，指定形状和数据数组。
   *
   * @param data Tensor 元素的浮点数数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(float[] data, long[] shape) {
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 创建一个新的 Tensor 实例，数据类型为 torch.int64，指定形状和数据数组。
   *
   * @param data Tensor 元素的长整数数组
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(long[] data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final LongBuffer longBuffer = allocateLongBuffer((int) numel(shape));
    // 将数据放入长整型缓冲区中
    longBuffer.put(data);
    // 使用长整型缓冲区和给定的形状创建一个新的 int64 数据类型的 Tensor 实例
    return new Tensor_int64(longBuffer, shape, memoryFormat);
  }

  /**
   * 使用指定形状和数据作为双精度浮点数数组创建一个新的 Tensor 实例，数据类型为 torch.float64。
   *
   * @param shape Tensor 的形状
   * @param data Tensor 的元素数据
   */
  public static Tensor fromBlob(double[] data, long[] shape, MemoryFormat memoryFormat) {
    // 检查数据数组不为空
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    // 检查形状数组不为空
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状的有效性
    checkShape(shape);
    // 检查数据数组长度与形状的一致性
    checkShapeAndDataCapacityConsistency(data.length, shape);
    // 分配一个双精度浮点数缓冲区
    final DoubleBuffer doubleBuffer = allocateDoubleBuffer((int) numel(shape));
    // 将数据放入双精度浮点数缓冲区中
    doubleBuffer.put(data);
    // 使用双精度浮点数缓冲区和给定的形状创建一个新的 float64 数据类型的 Tensor 实例
    return new Tensor_float64(doubleBuffer, shape, memoryFormat);
  }

  /**
   * 使用指定形状和数据作为无符号字节缓冲区创建一个新的 Tensor 实例，数据类型为 torch.uint8。
   *
   * @param data 直接缓冲区，使用本机字节顺序，包含 {@code Tensor.numel(shape)} 个元素。
   *             该缓冲区直接使用，对其内容的更改将更改张量。
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlobUnsigned(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    // 检查数据缓冲区不为空
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    // 检查形状数组不为空
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状的有效性
    checkShape(shape);
    // 检查数据缓冲区容量与形状的一致性
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    // 检查数据缓冲区必须是直接缓冲区
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    // 检查数据缓冲区必须具有本机字节顺序
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    // 使用无符号字节缓冲区和给定的形状创建一个新的 uint8 数据类型的 Tensor 实例
    return new Tensor_uint8(data, shape, memoryFormat);
  }

  /**
   * 使用指定形状和数据作为有符号字节缓冲区创建一个新的 Tensor 实例，数据类型为 torch.int8。
   *
   * @param data 直接缓冲区，使用本机字节顺序，包含 {@code Tensor.numel(shape)} 个元素。
   *             该缓冲区直接使用，对其内容的更改将更改张量。
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    // 检查数据缓冲区不为空
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    // 检查形状数组不为空
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状的有效性
    checkShape(shape);
    // 检查数据缓冲区容量与形状的一致性
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    // 检查数据缓冲区必须是直接缓冲区
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    // 检查数据缓冲区必须具有本机字节顺序
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    // 使用有符号字节缓冲区和给定的形状创建一个新的 int8 数据类型的 Tensor 实例
    return new Tensor_int8(data, shape, memoryFormat);
  }
  /**
   * 创建一个具有 torch.int8 数据类型、指定形状和数据的新 Tensor 实例。
   *
   * @param data 包含 {@code Tensor.numel(shape)} 元素的直接缓冲区，使用本地字节顺序，内容的更改将更改张量。
   * @param shape Tensor 的形状
   * @param memoryFormat 内存格式，指定张量的存储方式
   */
  public static Tensor fromBlob(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int8(data, shape, memoryFormat);
  }

  /**
   * 创建一个具有 torch.int32 数据类型、指定形状和数据的新 Tensor 实例。
   *
   * @param data 包含 {@code Tensor.numel(shape)} 元素的直接缓冲区，使用本地字节顺序，内容的更改将更改张量。
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(IntBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, shape, memoryFormat);
  }

  /**
   * 创建一个具有 torch.float32 数据类型、指定形状和数据的新 Tensor 实例。
   *
   * @param data 包含 {@code Tensor.numel(shape)} 元素的直接缓冲区，使用本地字节顺序，内容的更改将更改张量。
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(FloatBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, shape, memoryFormat);
  }

  /**
   * 创建一个具有 torch.int64 数据类型、指定形状和数据的新 Tensor 实例。
   *
   * @param data 包含 {@code Tensor.numel(shape)} 元素的直接缓冲区，使用本地字节顺序，内容的更改将更改张量。
   * @param shape Tensor 的形状
   */
  public static Tensor fromBlob(LongBuffer data, long[] shape, MemoryFormat memoryFormat) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    // 检查数据是否为直接缓冲区，否则抛出异常，提示数据缓冲区必须是直接的
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    // 检查数据缓冲区的字节顺序是否为本地字节顺序，否则抛出异常，提示数据缓冲区必须具有本地字节顺序
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    // 创建一个新的 Tensor_int64 实例，使用给定的 data、shape 和 memoryFormat
    return new Tensor_int64(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(LongBuffer data, long[] shape) {
    // 调用带有默认内存格式的 fromBlob 方法
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 使用指定的形状和数据创建一个新的 dtype 为 torch.float64 的 Tensor 实例。
   *
   * @param data 包含 {@code Tensor.numel(shape)} 元素的直接缓冲区，必须具有本地字节顺序。此缓冲区将直接使用，其内容的更改将更改张量。
   * @param shape Tensor 的形状
   * @param memoryFormat 张量的内存格式
   */
  public static Tensor fromBlob(DoubleBuffer data, long[] shape, MemoryFormat memoryFormat) {
    // 检查 data 缓冲区不为 null，否则抛出异常，提示数据缓冲区不能为空
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    // 检查 shape 不为 null，否则抛出异常，提示形状不能为空
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    // 检查形状的有效性
    checkShape(shape);
    // 检查数据容量和形状的一致性
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    // 检查数据是否为直接缓冲区，否则抛出异常，提示数据缓冲区必须是直接的
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    // 检查数据缓冲区的字节顺序是否为本地字节顺序，否则抛出异常，提示数据缓冲区必须具有本地字节顺序
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    // 创建一个新的 Tensor_float64 实例，使用给定的 data、shape 和 memoryFormat
    return new Tensor_float64(data, shape, memoryFormat);
  }

  public static Tensor fromBlob(DoubleBuffer data, long[] shape) {
    // 调用带有默认内存格式的 fromBlob 方法
    return fromBlob(data, shape, MemoryFormat.CONTIGUOUS);
  }

  @DoNotStrip private HybridData mHybridData;

  private Tensor(long[] shape, MemoryFormat memoryFormat) {
    // 检查形状的有效性
    checkShape(shape);
    // 复制形状数组，确保实例的形状是一个新副本
    this.shape = Arrays.copyOf(shape, shape.length);
    // 设置实例的内存格式
    this.memoryFormat = memoryFormat;
  }

  /** 返回张量中的元素数量。 */
  public long numel() {
    // 调用静态方法 numel 计算具有指定形状的张量的元素数量
    return numel(this.shape);
  }

  /** 计算具有指定形状的张量的元素数量。 */
  public static long numel(long[] shape) {
    // 检查形状的有效性
    checkShape(shape);
    // 初始化结果为1，计算元素数量
    int result = 1;
    for (long s : shape) {
      result *= s;
    }
    return result;
  }

  /** 返回张量的形状。（返回的数组是形状的新副本。） */
  public long[] shape() {
    // 返回形状的副本，以确保不可变性
    return Arrays.copyOf(shape, shape.length);
  }

  /** 返回张量的内存格式。 */
  public MemoryFormat memoryFormat() {
    // 返回实例的内存格式
    return memoryFormat;
  }

  /** @return 张量的数据类型。 */
  public abstract DType dtype();

  // 从本地调用
  @DoNotStrip
  int dtypeJniCode() {
    // 返回数据类型的 JNI 代码
    return dtype().jniCode;
  }

  // 从本地调用
  @DoNotStrip
  int memoryFormatJniCode() {
    // 返回内存格式的 JNI 代码
    return memoryFormat.jniCode;
  }

  /**
   * @return 包含张量数据的 Java 字节数组。这可能是一个拷贝或引用。
   * @throws IllegalStateException 如果对非 int8 张量调用此方法。
   */
  public byte[] getDataAsByteArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
}

/**
 * @return a Java byte array that contains the tensor data. This may be a copy or reference.
 * @throws IllegalStateException if it is called for a non-uint8 tensor.
 */
public byte[] getDataAsUnsignedByteArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
}

/**
 * @return a Java int array that contains the tensor data. This may be a copy or reference.
 * @throws IllegalStateException if it is called for a non-int32 tensor.
 */
public int[] getDataAsIntArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as int array.");
}

/**
 * @return a Java float array that contains the tensor data. This may be a copy or reference.
 * @throws IllegalStateException if it is called for a non-float32 tensor.
 */
public float[] getDataAsFloatArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
}

/**
 * @return a Java long array that contains the tensor data. This may be a copy or reference.
 * @throws IllegalStateException if it is called for a non-int64 tensor.
 */
public long[] getDataAsLongArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as long array.");
}

/**
 * @return a Java double array that contains the tensor data. This may be a copy or reference.
 * @throws IllegalStateException if it is called for a non-float64 tensor.
 */
public double[] getDataAsDoubleArray() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot return data as double array.");
}

@DoNotStrip
Buffer getRawDataBuffer() {
  throw new IllegalStateException(
      "Tensor of type " + getClass().getSimpleName() + " cannot " + "return raw data buffer.");
}

static class Tensor_uint8 extends Tensor {
  private final ByteBuffer data;

  private Tensor_uint8(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
    super(shape, memoryFormat);
    this.data = data;
  }

  @Override
  public DType dtype() {
    return DType.UINT8;
  }

  @Override
  Buffer getRawDataBuffer() {
    return data;
  }

  /**
   * @return a Java byte array that contains the tensor data. This may be a copy or reference.
   */
  @Override
  public byte[] getDataAsUnsignedByteArray() {
    // Rewind the ByteBuffer to start reading from the beginning
    data.rewind();
    // Create a byte array and read data from ByteBuffer into it
    byte[] arr = new byte[data.remaining()];
    data.get(arr);
    return arr;
  }

  @Override
  public String toString() {
    return String.format("Tensor(%s, dtype=torch.uint8)", Arrays.toString(shape));
  }
}

static class Tensor_int8 extends Tensor {
  private final ByteBuffer data;
  
  // Class implementation continues...
    private Tensor_int8(ByteBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }
    // 构造函数，初始化一个 int8 类型的 Tensor 对象
    // 参数包括数据缓冲区 data、形状 shape 和内存格式 memoryFormat

    @Override
    public DType dtype() {
      return DType.INT8;
    }
    // 返回当前 Tensor 对象的数据类型，这里是 INT8 类型

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }
    // 返回原始数据的缓冲区对象

    @Override
    public byte[] getDataAsByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }
    // 获取数据作为字节数组的形式
    // 首先将数据缓冲区重新定位到起始位置（rewind）
    // 然后创建一个与缓冲区大小相等的字节数组，并将数据读入该数组中返回

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int8)", Arrays.toString(shape));
    }
    // 返回 Tensor 对象的字符串表示形式
    // 格式为 "Tensor(形状数组, dtype=torch.int8)"
  }

  static class Tensor_int32 extends Tensor {
    private final IntBuffer data;

    private Tensor_int32(IntBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }
    // 构造函数，初始化一个 int32 类型的 Tensor 对象
    // 参数包括数据缓冲区 data、形状 shape 和内存格式 memoryFormat

    @Override
    public DType dtype() {
      return DType.INT32;
    }
    // 返回当前 Tensor 对象的数据类型，这里是 INT32 类型

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }
    // 返回原始数据的缓冲区对象

    @Override
    public int[] getDataAsIntArray() {
      data.rewind();
      int[] arr = new int[data.remaining()];
      data.get(arr);
      return arr;
    }
    // 获取数据作为整数数组的形式
    // 首先将数据缓冲区重新定位到起始位置（rewind）
    // 然后创建一个与缓冲区大小相等的整数数组，并将数据读入该数组中返回

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int32)", Arrays.toString(shape));
    }
    // 返回 Tensor 对象的字符串表示形式
    // 格式为 "Tensor(形状数组, dtype=torch.int32)"
  }

  static class Tensor_float32 extends Tensor {
    private final FloatBuffer data;

    Tensor_float32(FloatBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }
    // 构造函数，初始化一个 float32 类型的 Tensor 对象
    // 参数包括数据缓冲区 data、形状 shape 和内存格式 memoryFormat

    @Override
    public float[] getDataAsFloatArray() {
      data.rewind();
      float[] arr = new float[data.remaining()];
      data.get(arr);
      return arr;
    }
    // 获取数据作为浮点数数组的形式
    // 首先将数据缓冲区重新定位到起始位置（rewind）
    // 然后创建一个与缓冲区大小相等的浮点数数组，并将数据读入该数组中返回

    @Override
    public DType dtype() {
      return DType.FLOAT32;
    }
    // 返回当前 Tensor 对象的数据类型，这里是 FLOAT32 类型

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }
    // 返回原始数据的缓冲区对象

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float32)", Arrays.toString(shape));
    }
    // 返回 Tensor 对象的字符串表示形式
    // 格式为 "Tensor(形状数组, dtype=torch.float32)"
  }

  static class Tensor_int64 extends Tensor {
    private final LongBuffer data;

    private Tensor_int64(LongBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }
    // 构造函数，初始化一个 int64 类型的 Tensor 对象
    // 参数包括数据缓冲区 data、形状 shape 和内存格式 memoryFormat

    @Override
    public DType dtype() {
      return DType.INT64;
    }
    // 返回当前 Tensor 对象的数据类型，这里是 INT64 类型

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }
    // 返回原始数据的缓冲区对象

    @Override
    public long[] getDataAsLongArray() {
      data.rewind();
      long[] arr = new long[data.remaining()];
      data.get(arr);
      return arr;
    }
    // 获取数据作为长整数数组的形式
    // 首先将数据缓冲区重新定位到起始位置（rewind）
    // 然后创建一个与缓冲区大小相等的长整数数组，并将数据读入该数组中返回

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int64)", Arrays.toString(shape));
    }
    // 返回 Tensor 对象的字符串表示形式
    // 格式为 "Tensor(形状数组, dtype=torch.int64)"
  }

  static class Tensor_float64 extends Tensor {
    private final DoubleBuffer data;

    private Tensor_float64(DoubleBuffer data, long[] shape, MemoryFormat memoryFormat) {
      super(shape, memoryFormat);
      this.data = data;
    }
    // 构造函数，初始化一个 float64 类型的 Tensor 对象
    // 参数包括数据缓冲区 data、形状 shape 和内存格式 memoryFormat

    @Override
    public DType dtype() {
      return DType.FLOAT64;
    }
    // 返回当前 Tensor 对象的数据类型，这里是 FLOAT64 类型

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }
    // 返回原始数据的缓冲区对象

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float64)", Arrays.toString(shape));
    }
    // 返回 Tensor 对象的字符串表示形式
    // 格式为 "Tensor(形状数组, dtype=torch.float64)"
  }
  Buffer getRawDataBuffer() {
    return data;
  }

  @Override
  public double[] getDataAsDoubleArray() {
    // 将数据缓冲区重新定位到起始位置
    data.rewind();
    // 创建一个 double 数组，其大小与数据缓冲区剩余元素数量相同
    double[] arr = new double[data.remaining()];
    // 从数据缓冲区读取数据到 double 数组中
    data.get(arr);
    return arr;
  }

  @Override
  public String toString() {
    // 返回描述张量对象的字符串，包括形状信息和数据类型
    return String.format("Tensor(%s, dtype=torch.float64)", Arrays.toString(shape));
  }
}

// region checks
private static void checkArgument(boolean expression, String errorMessage, Object... args) {
  // 如果表达式为 false，则抛出带有格式化错误消息的 IllegalArgumentException 异常
  if (!expression) {
    throw new IllegalArgumentException(String.format(Locale.US, errorMessage, args));
  }
}

private static void checkShape(long[] shape) {
  // 检查形状数组不为 null
  checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
  // 遍历形状数组，确保每个维度的大小不为负数
  for (int i = 0; i < shape.length; i++) {
    checkArgument(shape[i] >= 0, ERROR_MSG_SHAPE_NON_NEGATIVE);
  }
}

private static void checkShapeAndDataCapacityConsistency(int dataCapacity, long[] shape) {
  // 计算形状数组表示的元素总数
  final long numel = numel(shape);
  // 检查数据容量与形状数组表示的元素总数是否一致，否则抛出异常
  checkArgument(
      numel == dataCapacity,
      "Inconsistent data capacity:%d and shape number elements:%d shape:%s",
      dataCapacity,
      numel,
      Arrays.toString(shape));
}
// endregion checks

// Called from native
@DoNotStrip
private static Tensor nativeNewTensor(
    ByteBuffer data, long[] shape, int dtype, int memoryFormatCode, HybridData hybridData) {
  Tensor tensor = null;

  MemoryFormat memoryFormat = MemoryFormat.CONTIGUOUS;
  // 根据内存格式代码确定张量的内存格式
  if (MemoryFormat.CHANNELS_LAST.jniCode == memoryFormatCode) {
    memoryFormat = MemoryFormat.CHANNELS_LAST;
  } else if (MemoryFormat.CHANNELS_LAST_3D.jniCode == memoryFormatCode) {
    memoryFormat = MemoryFormat.CHANNELS_LAST_3D;
  }

  // 根据数据类型代码创建相应类型的张量对象
  if (DType.FLOAT32.jniCode == dtype) {
    tensor = new Tensor_float32(data.asFloatBuffer(), shape, memoryFormat);
  } else if (DType.INT32.jniCode == dtype) {
    tensor = new Tensor_int32(data.asIntBuffer(), shape, memoryFormat);
  } else if (DType.INT64.jniCode == dtype) {
    tensor = new Tensor_int64(data.asLongBuffer(), shape, memoryFormat);
  } else if (DType.FLOAT64.jniCode == dtype) {
    tensor = new Tensor_float64(data.asDoubleBuffer(), shape, memoryFormat);
  } else if (DType.UINT8.jniCode == dtype) {
    tensor = new Tensor_uint8(data, shape, memoryFormat);
  } else if (DType.INT8.jniCode == dtype) {
    tensor = new Tensor_int8(data, shape, memoryFormat);
  } else {
    // 如果未知数据类型代码，抛出异常
    new IllegalArgumentException("Unknown Tensor dtype");
  }
  // 将原生数据对象关联到张量对象
  tensor.mHybridData = hybridData;
  return tensor;
}
}
```