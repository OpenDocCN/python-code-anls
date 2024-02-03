# `.\PaddleOCR\deploy\cpp_infer\src\preprocess_op.cpp`

```
// 包含版权声明和许可信息

#include <include/preprocess_op.h>  // 包含预处理操作的头文件

namespace PaddleOCR {

void Permute::Run(const cv::Mat *im, float *data) {  // 对图像进行通道排列
  int rh = im->rows;  // 获取图像的行数
  int rw = im->cols;  // 获取图像的列数
  int rc = im->channels();  // 获取图像的通道数
  for (int i = 0; i < rc; ++i) {  // 遍历每个通道
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);  // 提取通道数据
  }
}

void PermuteBatch::Run(const std::vector<cv::Mat> imgs, float *data) {  // 对图像批量进行通道排列
  for (int j = 0; j < imgs.size(); j++) {  // 遍历每张图像
    int rh = imgs[j].rows;  // 获取图像的行数
    int rw = imgs[j].cols;  // 获取图像的列数
    int rc = imgs[j].channels();  // 获取图像的通道数
    for (int i = 0; i < rc; ++i) {  // 遍历每个通道
      cv::extractChannel(
          imgs[j], cv::Mat(rh, rw, CV_32FC1, data + (j * rc + i) * rh * rw), i);  // 提取通道数据
    }
  }
}

void Normalize::Run(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &scale, const bool is_scale) {  // 对图像进行归一化
  double e = 1.0;  // 初始化缩放因子
  if (is_scale) {  // 如果需要缩放
    e /= 255.0;  // 更新缩放因子
  }
  (*im).convertTo(*im, CV_32FC3, e);  // 将图像转换为32位浮点数格式
  std::vector<cv::Mat> bgr_channels(3);  // 创建三个通道的图像
  cv::split(*im, bgr_channels);  // 分离图像通道
  for (auto i = 0; i < bgr_channels.size(); i++) {  // 遍历每个通道
    bgr_channels[i].convertTo(bgr_channels[i], CV_32FC1, 1.0 * scale[i],
                              (0.0 - mean[i]) * scale[i]);  // 归一化通道数据
  }
  cv::merge(bgr_channels, *im);  // 合并通道数据
}
void ResizeImgType0::Run(const cv::Mat &img, cv::Mat &resize_img,
                         std::string limit_type, int limit_side_len,
                         float &ratio_h, float &ratio_w, bool use_tensorrt) {
  // 获取输入图像的宽度和高度
  int w = img.cols;
  int h = img.rows;
  // 初始化缩放比例为1
  float ratio = 1.f;
  // 根据限制类型进行缩放比例计算
  if (limit_type == "min") {
    // 计算最小宽高
    int min_wh = std::min(h, w);
    // 如果最小宽高小于限制边长
    if (min_wh < limit_side_len) {
      // 根据高宽比例计算缩放比例
      if (h < w) {
        ratio = float(limit_side_len) / float(h);
      } else {
        ratio = float(limit_side_len) / float(w);
      }
    }
  } else {
    // 计算最大宽高
    int max_wh = std::max(h, w);
    // 如果最大宽高大于限制边长
    if (max_wh > limit_side_len) {
      // 根据高宽比例计算缩放比例
      if (h > w) {
        ratio = float(limit_side_len) / float(h);
      } else {
        ratio = float(limit_side_len) / float(w);
      }
    }
  }

  // 根据缩放比例计算新的高度和宽度
  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);

  // 对新的高度和宽度进行调整，使其为32的倍数
  resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);

  // 使用插值方法对图像进行缩放
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  // 计算高度和宽度的缩放比例
  ratio_h = float(resize_h) / float(h);
  ratio_w = float(resize_w) / float(w);
}

void CrnnResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                        bool use_tensorrt,
                        const std::vector<int> &rec_image_shape) {
  // 获取图像通道数、高度和宽度
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  // 根据宽高比例计算新的宽度
  imgW = int(imgH * wh_ratio);

  // 计算图像的高宽比例
  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;

  // 根据高宽比例计算新的宽度
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  // 使用插值方法对图像进行缩放
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  // 在图像边界填充0值，使其宽度与指定宽度一致
  cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                     int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                     {0, 0, 0});
}
void ClsResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                       bool use_tensorrt,
                       const std::vector<int> &rec_image_shape) {
  // 定义变量 imgC, imgH, imgW，用于存储图像通道数、高度和宽度
  int imgC, imgH, imgW;
  // 从 rec_image_shape 中获取图像通道数、高度和宽度
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  // 计算图像宽高比
  float ratio = float(img.cols) / float(img.rows);
  int resize_w, resize_h;
  // 根据比例调整图像宽度
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  // 调整图像大小
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
}

void TableResizeImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                         const int max_len) {
  // 获取图像宽度和高度
  int w = img.cols;
  int h = img.rows;

  // 获取宽高中的最大值
  int max_wh = w >= h ? w : h;
  // 计算调整比例
  float ratio = w >= h ? float(max_len) / float(w) : float(max_len) / float(h);

  // 根据比例调整图像高度和宽度
  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);

  // 调整图像大小
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
}

void TablePadImg::Run(const cv::Mat &img, cv::Mat &resize_img,
                      const int max_len) {
  // 获取图像宽度和高度
  int w = img.cols;
  int h = img.rows;
  // 在图像周围填充边框
  cv::copyMakeBorder(img, resize_img, 0, max_len - h, 0, max_len - w,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void Resize::Run(const cv::Mat &img, cv::Mat &resize_img, const int h,
                 const int w) {
  // 调整图像大小
  cv::resize(img, resize_img, cv::Size(w, h));
}

} // namespace PaddleOCR
```