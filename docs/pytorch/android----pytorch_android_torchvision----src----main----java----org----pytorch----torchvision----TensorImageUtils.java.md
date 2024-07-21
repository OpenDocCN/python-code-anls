# `.\pytorch\android\pytorch_android_torchvision\src\main\java\org\pytorch\torchvision\TensorImageUtils.java`

```
/**
 * Contains utility functions for creating {@link org.pytorch.Tensor} instances from
 * {@link android.graphics.Bitmap} or {@link android.media.Image} sources.
 */
public final class TensorImageUtils {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

  /**
   * Creates a new {@link org.pytorch.Tensor} from a full {@link android.graphics.Bitmap},
   * normalizing it with the specified mean and standard deviation.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB order
   * @param memoryFormat memory layout format for the tensor
   * @return a new Tensor containing the bitmap data
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      final float[] normMeanRGB,
      final float[] normStdRGB,
      final MemoryFormat memoryFormat) {
    // Check validity of normalization mean values
    checkNormMeanArg(normMeanRGB);
    // Check validity of normalization standard deviation values
    checkNormStdArg(normStdRGB);

    // Delegate to overloaded method to process entire bitmap
    return bitmapToFloat32Tensor(
        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB, memoryFormat);
  }

  /**
   * Creates a new {@link org.pytorch.Tensor} from a {@link android.graphics.Bitmap},
   * normalizing it with the specified mean and standard deviation.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB order
   * @return a new Tensor containing the bitmap data
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap, final float[] normMeanRGB, final float[] normStdRGB) {
    // Delegate to the full method with default memory format
    return bitmapToFloat32Tensor(
        bitmap,
        0,
        0,
        bitmap.getWidth(),
        bitmap.getHeight(),
        normMeanRGB,
        normStdRGB,
        MemoryFormat.CONTIGUOUS);
  }

  /**
   * Writes tensor content from a specified {@link android.graphics.Bitmap},
   * normalizing it with specified mean and std, to a {@link java.nio.FloatBuffer}
   * at a specified offset.
   *
   * @param bitmap {@link android.graphics.Bitmap} as the source for Tensor data
   * @param x x coordinate of the top-left corner of the bitmap area
   * @param y y coordinate of the top-left corner of the bitmap area
   * @param width width of the bitmap area
   * @param height height of the bitmap area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB order
   * @param outBuffer destination {@link java.nio.FloatBuffer} to write the tensor data
   * @param outBufferOffset offset within the outBuffer where writing starts
   * @param memoryFormat memory layout format for the tensor
   */
  public static void bitmapToFloatBuffer(
      final Bitmap bitmap,
      final int x,
      final int y,
      final int width,
      final int height,
      final float[] normMeanRGB,
      final float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset,
      final MemoryFormat memoryFormat) {
    // Check if the output buffer has sufficient capacity for the specified bitmap area
    checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
    // Method implementation continues beyond this point
    // 检查并确保 normMeanRGB 数组长度为 3
    checkNormMeanArg(normMeanRGB);
    // 检查并确保 normStdRGB 数组长度为 3
    checkNormStdArg(normStdRGB);
    // 如果内存格式不是连续的或者通道在最后的格式，抛出不支持的内存格式异常
    if (memoryFormat != MemoryFormat.CONTIGUOUS && memoryFormat != MemoryFormat.CHANNELS_LAST) {
      throw new IllegalArgumentException("Unsupported memory format " + memoryFormat);
    }

    // 计算像素总数
    final int pixelsCount = height * width;
    // 创建用于存储像素的整数数组
    final int[] pixels = new int[pixelsCount];
    // 从位图中获取像素数据并存储到 pixels 数组中
    bitmap.getPixels(pixels, 0, width, x, y, width, height);

    // 如果内存格式为连续
    if (MemoryFormat.CONTIGUOUS == memoryFormat) {
      // 计算绿色通道的偏移量
      final int offset_g = pixelsCount;
      // 计算蓝色通道的偏移量
      final int offset_b = 2 * pixelsCount;
      // 遍历像素数组
      for (int i = 0; i < pixelsCount; i++) {
        // 获取当前像素的颜色值
        final int c = pixels[i];
        // 提取红色通道，并进行归一化
        float r = ((c >> 16) & 0xff) / 255.0f;
        // 提取绿色通道，并进行归一化
        float g = ((c >> 8) & 0xff) / 255.0f;
        // 提取蓝色通道，并进行归一化
        float b = ((c) & 0xff) / 255.0f;
        // 将归一化后的值写入输出缓冲区，考虑均值和标准差
        outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
        outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
        outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
      }
    } else {
      // 如果内存格式不是连续
      // 遍历像素数组
      for (int i = 0; i < pixelsCount; i++) {
        // 获取当前像素的颜色值
        final int c = pixels[i];
        // 提取红色通道，并进行归一化
        float r = ((c >> 16) & 0xff) / 255.0f;
        // 提取绿色通道，并进行归一化
        float g = ((c >> 8) & 0xff) / 255.0f;
        // 提取蓝色通道，并进行归一化
        float b = ((c) & 0xff) / 255.0f;
        // 将归一化后的值写入输出缓冲区，考虑均值和标准差，按照非连续的内存布局写入
        outBuffer.put(outBufferOffset + 3 * i + 0, (r - normMeanRGB[0]) / normStdRGB[0]);
        outBuffer.put(outBufferOffset + 3 * i + 1, (g - normMeanRGB[1]) / normStdRGB[1]);
        outBuffer.put(outBufferOffset + 3 * i + 2, (b - normMeanRGB[2]) / normStdRGB[2]);
      }
    }
  }

  /**
   * 将 {@link android.graphics.Bitmap} 转换为 {@link org.pytorch.Tensor}，从指定区域提取，并使用指定的均值和标准差进行归一化。
   *
   * @param bitmap {@link android.graphics.Bitmap} 作为 Tensor 数据源
   * @param x - 位图区域左上角的 x 坐标
   * @param y - 位图区域左上角的 y 坐标
   * @param width - 位图区域的宽度
   * @param height - 位图区域的高度
   * @param normMeanRGB RGB 通道的均值用于归一化，长度必须为 3，顺序为 RGB
   * @param normStdRGB RGB 通道的标准差用于归一化，长度必须为 3，顺序为 RGB
   * @param memoryFormat 内存格式，指定输出数据的布局方式
   * @return 新的 {@link org.pytorch.Tensor} 对象
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB,
      MemoryFormat memoryFormat) {
    // 检查均值参数的有效性
    checkNormMeanArg(normMeanRGB);
    // 检查和规范化 normStdRGB 参数
    checkNormStdArg(normStdRGB);

    // 创建一个可以容纳指定大小数据的浮点数缓冲区
    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    
    // 将 Bitmap 转换为浮点数缓冲区，应用归一化的均值和标准差
    bitmapToFloatBuffer(
        bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0, memoryFormat);
    
    // 将浮点数缓冲区转换为 Tensor 对象，并指定其形状和内存布局
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, height, width}, memoryFormat);
  }

  /**
   * 将 Bitmap 转换为 Float32 格式的 Tensor 对象，使用默认的内存布局 CONTIGUOUS。
   *
   * @param bitmap {@link android.graphics.Bitmap} 作为 Tensor 数据源
   * @param x 起始 x 坐标
   * @param y 起始 y 坐标
   * @param width 返回的 Tensor 宽度，必须为正数
   * @param height 返回的 Tensor 高度，必须为正数
   * @param normMeanRGB RGB 通道的均值数组，长度必须为 3，顺序为 RGB
   * @param normStdRGB RGB 通道的标准差数组，长度必须为 3，顺序为 RGB
   * @return 转换后的 Tensor 对象
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB) {
    // 调用带有默认内存布局参数的 bitmapToFloat32Tensor 方法
    return bitmapToFloat32Tensor(
        bitmap, x, y, width, height, normMeanRGB, normStdRGB, MemoryFormat.CONTIGUOUS);
  }

  /**
   * 从指定区域的 {@link android.media.Image} 创建新的 {@link org.pytorch.Tensor}，
   * 可进行可选的旋转、缩放（最近邻）和中心裁剪。
   *
   * @param image {@link android.media.Image} 作为 Tensor 数据源
   * @param rotateCWDegrees 需要顺时针旋转输入图像使其直立的角度。有效值范围：0、90、180、270
   * @param tensorWidth 返回的 Tensor 宽度，必须为正数
   * @param tensorHeight 返回的 Tensor 高度，必须为正数
   * @param normMeanRGB RGB 通道的均值数组，长度必须为 3，顺序为 RGB
   * @param normStdRGB RGB 通道的标准差数组，长度必须为 3，顺序为 RGB
   * @param memoryFormat Tensor 的内存布局格式
   * @return 转换后的 Tensor 对象
   */
  public static Tensor imageYUV420CenterCropToFloat32Tensor(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      MemoryFormat memoryFormat) {
    // 如果 Image 的格式不是 YUV_420_888，则抛出异常
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    // 检查和规范化 normMeanRGB 参数
    checkNormMeanArg(normMeanRGB);
    
    // 检查和规范化 normStdRGB 参数
    checkNormStdArg(normStdRGB);
    
    // 检查旋转角度是否有效
    checkRotateCWDegrees(rotateCWDegrees);
    
    // 检查返回的 Tensor 大小是否有效
    checkTensorSize(tensorWidth, tensorHeight);

    // 创建一个可以容纳指定大小数据的浮点数缓冲区
    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * tensorWidth * tensorHeight);
    
    // 将 YUV_420_888 图像进行中心裁剪并转换为浮点数缓冲区，应用归一化的均值和标准差
    imageYUV420CenterCropToFloatBuffer(
        image,
        rotateCWDegrees,
        tensorWidth,
        tensorHeight,
        normMeanRGB,
        normStdRGB,
        floatBuffer,
        0,
        memoryFormat);
    
    // 将浮点数缓冲区转换为 Tensor 对象，并指定其形状和内存布局
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, tensorHeight, tensorWidth}, memoryFormat);
  }

  /**
   * 将 YUV_420_888 格式的 {@link android.media.Image} 进行中心裁剪并转换为 Float32 格式的 Tensor 对象，
   * 使用默认的内存布局 CONTIGUOUS。
   *
   * @param image {@link android.media.Image} 作为 Tensor 数据源
   * @param rotateCWDegrees 需要顺时针旋转输入图像使其直立的角度。有效值范围：0、90、180、270
   * @param tensorWidth 返回的 Tensor 宽度，必须为正数
   * @param tensorHeight 返回的 Tensor 高度，必须为正数
   * @param normMeanRGB RGB 通道的均值数组，长度必须为 3，顺序为 RGB
   * @param normStdRGB RGB 通道的标准差数组，长度必须为 3，顺序为 RGB
   * @return 转换后的 Tensor 对象
   */
  public static Tensor imageYUV420CenterCropToFloat32Tensor(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB) {
    // 调用函数处理 YUV420 格式的图像，进行中心裁剪并转换为 float32 数据格式的张量，并返回结果
    return imageYUV420CenterCropToFloat32Tensor(
        image,                    // 输入的 android.media.Image 对象，作为张量数据的源
        rotateCWDegrees,          // 顺时针旋转角度，用于将输入图像旋转至正常方向，可选值为 0, 90, 180, 270
        tensorWidth,              // 返回的张量宽度，必须为正数
        tensorHeight,             // 返回的张量高度，必须为正数
        normMeanRGB,              // RGB 通道的均值数组，长度必须为 3，顺序为 RGB
        normStdRGB,               // RGB 通道的标准差数组，长度必须为 3，顺序为 RGB
        MemoryFormat.CONTIGUOUS   // 输出的内存格式，这里使用 CONTIGUOUS 表示连续存储
    );
  }

  /**
   * Writes tensor content from specified {@link android.media.Image}, doing optional rotation,
   * scaling (nearest) and center cropping to specified {@link java.nio.FloatBuffer} with specified
   * offset.
   *
   * @param image {@link android.media.Image} as a source for Tensor data
   * @param rotateCWDegrees Clockwise angle through which the input image needs to be rotated to be
   *     upright. Range of valid values: 0, 90, 180, 270
   * @param tensorWidth return tensor width, must be positive
   * @param tensorHeight return tensor height, must be positive
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   * @param outBuffer Output buffer, where tensor content will be written
   * @param outBufferOffset Output buffer offset with which tensor content will be written
   * @param memoryFormat Memory format for storing tensor content, either CONTIGUOUS or CHANNELS_LAST
   */
  public static void imageYUV420CenterCropToFloatBuffer(
      final Image image,
      int rotateCWDegrees,
      final int tensorWidth,
      final int tensorHeight,
      float[] normMeanRGB,
      float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset,
      final MemoryFormat memoryFormat) {
    // 检查输出缓冲区的容量是否足够存储给定尺寸的张量数据
    checkOutBufferCapacity(outBuffer, outBufferOffset, tensorWidth, tensorHeight);

    // 检查输入图像的格式是否为 YUV_420_888
    if (image.getFormat() != ImageFormat.YUV_420_888) {
      throw new IllegalArgumentException(
          String.format(
              Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
    }

    // 检查 RGB 均值数组的有效性
    checkNormMeanArg(normMeanRGB);
    // 检查 RGB 标准差数组的有效性
    checkNormStdArg(normStdRGB);
    // 检查旋转角度的有效性
    checkRotateCWDegrees(rotateCWDegrees);
    // 检查张量尺寸的有效性
    checkTensorSize(tensorWidth, tensorHeight);

    // 获取图像的各个平面数据
    Image.Plane[] planes = image.getPlanes();
    Image.Plane Y = planes[0];
    Image.Plane U = planes[1];
    Image.Plane V = planes[2];

    // 根据指定的 MemoryFormat 确定对应的 JNI 代码值
    int memoryFormatJniCode = 0;
    if (MemoryFormat.CONTIGUOUS == memoryFormat) {
      memoryFormatJniCode = 1;
    } else if (MemoryFormat.CHANNELS_LAST == memoryFormat) {
      memoryFormatJniCode = 2;
    }
    # 调用本地方法，将 YUV420 格式的图像中心裁剪并转换为浮点缓冲区
    NativePeer.imageYUV420CenterCropToFloatBuffer(
        Y.getBuffer(),           # 获取 Y 通道的 ByteBuffer
        Y.getRowStride(),        # 获取 Y 通道的行跨度
        Y.getPixelStride(),      # 获取 Y 通道的像素跨度
        U.getBuffer(),           # 获取 U 通道的 ByteBuffer
        V.getBuffer(),           # 获取 V 通道的 ByteBuffer
        U.getRowStride(),        # 获取 U 通道的行跨度
        U.getPixelStride(),      # 获取 U 通道的像素跨度
        image.getWidth(),        # 获取图像宽度
        image.getHeight(),       # 获取图像高度
        rotateCWDegrees,         # 顺时针旋转角度
        tensorWidth,             # 张量宽度
        tensorHeight,            # 张量高度
        normMeanRGB,             # RGB 像素的均值归一化数组
        normStdRGB,              # RGB 像素的标准差归一化数组
        outBuffer,               # 输出的浮点缓冲区
        outBufferOffset,         # 输出缓冲区的偏移量
        memoryFormatJniCode      # 内存格式的 JNI 代码
    )
  }
}



# 这行代码表示一个代码块的结束，与之对应的应该是一个代码块的开始或者一个条件语句的结束。
```