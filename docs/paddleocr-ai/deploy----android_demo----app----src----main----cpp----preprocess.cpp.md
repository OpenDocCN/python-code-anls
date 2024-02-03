# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\preprocess.cpp`

```
// 将 Android Bitmap 转换为 OpenCV Mat 对象
cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap) {
  // 定义 Android Bitmap 信息结构体
  AndroidBitmapInfo info;
  // 获取 Android Bitmap 的信息
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  // 如果获取信息失败，则记录错误信息并返回空的 Mat 对象
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_getInfo failed, result: %d", result);
    return cv::Mat{};
  }
  // 如果 Bitmap 格式不是 RGBA_8888，则记录错误信息并返回空的 Mat 对象
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Bitmap format is not RGBA_8888 !");
    return cv::Mat{};
  }
  // 锁定 Bitmap 像素数据
  unsigned char *srcData = NULL;
  AndroidBitmap_lockPixels(env, bitmap, (void **)&srcData);
  // 创建一个与 Bitmap 大小相同的 Mat 对象
  cv::Mat mat = cv::Mat::zeros(info.height, info.width, CV_8UC4);
  // 将 Bitmap 数据复制到 Mat 对象中
  memcpy(mat.data, srcData, info.height * info.width * 4);
  // 解锁 Bitmap 像素数据
  AndroidBitmap_unlockPixels(env, bitmap);
  // 将 RGBA 格式转换为 BGR 格式
  cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
  
  // 如果需要将 Mat 对象保存为图片文件，可以使用 imwrite 函数
  /**
  if (!cv::imwrite("/sdcard/1/copy.jpg", mat)){
      LOGE("Write image failed " );
  }
   */

  // 返回转换后的 Mat 对象
  return mat;
}

// 调整图像大小
cv::Mat resize_img(const cv::Mat &img, int height, int width) {
  // 如果图像大小已经符合要求，则直接返回原图像
  if (img.rows == height && img.cols == width) {
    return img;
  }
  // 创建一个新的 Mat 对象，调整大小为指定的高度和宽度
  cv::Mat new_img;
  cv::resize(img, new_img, cv::Size(height, width));
  // 返回调整大小后的新图像
  return new_img;
}

// 使用 NEON 指令集加速填充张量的均值和缩放，并转换布局：nhwc -> nchw
void neon_mean_scale(const float *din, float *dout, int size,
                     const std::vector<float> &mean,
                     const std::vector<float> &scale) {
  // 检查均值和缩放参数的大小是否正确
  if (mean.size() != 3 || scale.size() != 3) {
    LOGE("[ERROR] mean or scale size must equal to 3");
    return;
  }

  // 使用 NEON 指令加载均值和缩放参数
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  // 分别指向输出张量的三个通道
  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  // 使用 NEON 指令进行均值和缩放的计算
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    # 计算 vin3.val[1] 与 vmean1 的差值
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    # 计算 vin3.val[2] 与 vmean2 的差值
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    # 将 vsub0 与 vscale0 相乘
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    # 将 vsub1 与 vscale1 相乘
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    # 将 vsub2 与 vscale2 相乘
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    # 将 vs0 中的值存储到 dout_c0 中
    vst1q_f32(dout_c0, vs0);
    # 将 vs1 中的值存储到 dout_c1 中
    vst1q_f32(dout_c1, vs1);
    # 将 vs2 中的值存储到 dout_c2 中
    vst1q_f32(dout_c2, vs2);

    # din 指针向后移动 12 个位置
    din += 12;
    # dout_c0 指针向后移动 4 个位置
    dout_c0 += 4;
    # dout_c1 指针向后移动 4 个位置
    dout_c1 += 4;
    # dout_c2 指针向后移动 4 个位置
    dout_c2 += 4;
  }
  # 对于剩余的元素，逐个计算并存储到对应的输出指针中
  for (; i < size; i++) {
    # 计算并存储第一个通道的值
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    # 计算并存储第二个通道的值
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    # 计算并存储第三个通道的值
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
# 闭合大括号，表示代码块的结束
```