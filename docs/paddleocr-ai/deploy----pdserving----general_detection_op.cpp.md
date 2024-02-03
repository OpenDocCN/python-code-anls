# `.\PaddleOCR\deploy\pdserving\general_detection_op.cpp`

```py
// 版权声明和许可证信息
// 该代码受 Apache 许可证版本 2.0 保护
// 请在遵守许可证的情况下使用该文件
// 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
// 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
// 没有任何形式的担保或条件，无论是明示的还是暗示的
// 请查看许可证以获取特定语言的权限和限制

#include "core/general-server/op/general_detection_op.h"
#include "core/predictor/framework/infer.h"
#include "core/predictor/framework/memory.h"
#include "core/predictor/framework/resource.h"
#include "core/util/include/timer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

/*
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
*/

namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::Timer;
using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::predictor::general_model::Tensor;
using baidu::paddle_serving::predictor::general_model::Response;
using baidu::paddle_serving::predictor::general_model::Request;
using baidu::paddle_serving::predictor::InferManager;
using baidu::paddle_serving::predictor::PaddleGeneralModelConfig;

// 执行推理操作
int GeneralDetectionOp::inference() {
  // 输出日志信息
  VLOG(2) << "Going to run inference";
  // 获取前驱节点的名称
  const std::vector<std::string> pre_node_names = pre_names();
  // 检查前驱节点数量是否为1
  if (pre_node_names.size() != 1) {
    LOG(ERROR) << "This op(" << op_name()
               << ") can only have one predecessor op, but received "
               << pre_node_names.size();
    return -1;
  }
  // 获取前驱节点的名称
  const std::string pre_name = pre_node_names[0];

  // 获取前驱节点的输入数据
  const GeneralBlob *input_blob = get_depend_argument<GeneralBlob>(pre_name);
  // 如果输入数据为空
  if (!input_blob) {
    // 输出错误日志，如果输入的 input_blob 为空指针，则返回错误代码 -1
    LOG(ERROR) << "input_blob is nullptr,error";
    return -1;
  }
  // 获取输入 blob 的日志 ID
  uint64_t log_id = input_blob->GetLogId();
  // 输出日志信息，包括日志 ID 和先前操作的名称
  VLOG(2) << "(logid=" << log_id << ") Get precedent op name: " << pre_name;

  // 获取可变类型的 output_blob，如果为空指针则返回错误代码 -1
  GeneralBlob *output_blob = mutable_data<GeneralBlob>();
  if (!output_blob) {
    LOG(ERROR) << "output_blob is nullptr,error";
    return -1;
  }
  // 设置输出 blob 的日志 ID 为输入 blob 的日志 ID
  output_blob->SetLogId(log_id);

  // 如果输入 blob 为空指针，则输出错误日志并返回错误代码 -1
  if (!input_blob) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed mutable depended argument, op:" << pre_name;
    return -1;
  }

  // 获取输入和输出 blob 的张量向量
  const TensorVector *in = &input_blob->tensor_vector;
  TensorVector *out = &output_blob->tensor_vector;

  // 获取输入 blob 的批处理大小，并输出日志信息
  int batch_size = input_blob->_batch_size;
  VLOG(2) << "(logid=" << log_id << ") input batch size: " << batch_size;

  // 设置输出 blob 的批处理大小为输入 blob 的批处理大小
  output_blob->_batch_size = batch_size;

  // 初始化变量和指针
  std::vector<int> input_shape;
  int in_num = 0;
  void *databuf_data = NULL;
  char *databuf_char = NULL;
  size_t databuf_size = 0;
  // 目前仅支持单个字符串
  char *total_input_ptr = static_cast<char *>(in->at(0).data.data());
  std::string base64str = total_input_ptr;

  // 初始化变量并创建 OpenCV 的 Mat 对象
  float ratio_h{};
  float ratio_w{};
  cv::Mat img = Base2Mat(base64str);
  cv::Mat srcimg;
  cv::Mat resize_img;

  cv::Mat resize_img_rec;
  cv::Mat crop_img;
  // 将 img 复制到 srcimg
  img.copyTo(srcimg);

  // 运行 resize_op_ 操作，对 img 进行缩放处理
  this->resize_op_.Run(img, resize_img, this->max_side_len_, ratio_h, ratio_w,
                       this->use_tensorrt_);

  // 运行 normalize_op_ 操作，对 resize_img 进行归一化处理
  this->normalize_op_.Run(&resize_img, this->mean_det, this->scale_det,
                          this->is_scale_);

  // 初始化输入向量 input，并运行 permute_op_ 操作
  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());

  // 创建新的 TensorVector 对象 real_in，如果为空指针则返回错误代码 -1
  TensorVector *real_in = new TensorVector();
  if (!real_in) {
    LOG(ERROR) << "real_in is nullptr,error";
    return -1;
  }

  // 遍历输入张量向量，计算输入形状和数量
  for (int i = 0; i < in->size(); ++i) {
    input_shape = {1, 3, resize_img.rows, resize_img.cols};
    in_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                             std::multiplies<int>());
    // 计算数据缓冲区的大小，为输入数量乘以 float 类型的大小
    databuf_size = in_num * sizeof(float);
    // 从内存池中分配内存，大小为数据缓冲区的大小
    databuf_data = MempoolWrapper::instance().malloc(databuf_size);
    // 如果内存分配失败，则记录错误信息并返回 -1
    if (!databuf_data) {
      LOG(ERROR) << "Malloc failed, size: " << databuf_size;
      return -1;
    }
    // 将输入数据拷贝到数据缓冲区中
    memcpy(databuf_data, input.data(), databuf_size);
    // 将数据缓冲区转换为 char 类型
    databuf_char = reinterpret_cast<char *>(databuf_data);
    // 创建 PaddleBuf 对象，用于存储数据缓冲区的内容
    paddle::PaddleBuf paddleBuf(databuf_char, databuf_size);
    // 创建 PaddleTensor 对象，设置名称、数据类型、形状和数据
    paddle::PaddleTensor tensor_in;
    tensor_in.name = in->at(i).name;
    tensor_in.dtype = paddle::PaddleDType::FLOAT32;
    tensor_in.shape = {1, 3, resize_img.rows, resize_img.cols};
    tensor_in.lod = in->at(i).lod;
    tensor_in.data = paddleBuf;
    // 将 PaddleTensor 对象添加到 real_in 中
    real_in->push_back(tensor_in);
  }

  // 创建计时器对象
  Timer timeline;
  // 获取当前时间戳
  int64_t start = timeline.TimeStampUS();
  // 启动计时器
  timeline.Start();

  // 使用 InferManager 进行推理，如果失败则记录错误信息并返回 -1
  if (InferManager::instance().infer(engine_name().c_str(), real_in, out,
                                     batch_size)) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed do infer in fluid model: " << engine_name().c_str();
    return -1;
  }
  // 释放 real_in 对象的内存
  delete real_in;

  // 初始化输出形状和数量
  std::vector<int> output_shape;
  int out_num = 0;
  void *databuf_data_out = NULL;
  char *databuf_char_out = NULL;
  size_t databuf_size_out = 0;
  // 为 PaddleOCR 后处理添加特殊处理
  int infer_outnum = out->size();
  for (int k = 0; k < infer_outnum; ++k) {
    // 获取输出的形状信息
    int n2 = out->at(k).shape[2];
    int n3 = out->at(k).shape[3];
    int n = n2 * n3;

    // 获取输出数据，并进行后处理
    float *out_data = static_cast<float *>(out->at(k).data.data());
    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
      pred[i] = float(out_data[i]);
      cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    // 创建 OpenCV 的 Mat 对象
    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    // 设置阈值和最大值
    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    // 对输入的图像进行阈值处理，生成二值图像
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    
    // 创建膨胀操作所需的结构元素
    cv::Mat dilation_map;
    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    
    // 对二值图像进行膨胀操作
    cv::dilate(bit_map, dilation_map, dila_ele);
    
    // 从预测图和膨胀后的图像中获取文本框信息
    boxes = post_processor_.BoxesFromBitmap(pred_map, dilation_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_);
    
    // 过滤文本检测结果
    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
    
    // 初始化最大宽高比、裁剪图像、调整大小后的图像等变量
    float max_wh_ratio = 0.0f;
    std::vector<cv::Mat> crop_imgs;
    std::vector<cv::Mat> resize_imgs;
    int max_resize_w = 0;
    int max_resize_h = 0;
    int box_num = boxes.size();
    std::vector<std::vector<float>> output_rec;
    
    // 遍历每个文本框
    for (int i = 0; i < box_num; ++i) {
        // 获取旋转裁剪后的图像
        cv::Mat line_img = GetRotateCropImage(img, boxes[i]);
        // 计算宽高比
        float wh_ratio = float(line_img.cols) / float(line_img.rows);
        // 更新最大宽高比
        max_wh_ratio = max_wh_ratio > wh_ratio ? max_wh_ratio : wh_ratio;
        // 将裁剪后的图像添加到列表中
        crop_imgs.push_back(line_img);
    }
    
    // 遍历每个文本框
    for (int i = 0; i < box_num; ++i) {
        cv::Mat resize_img;
        // 获取裁剪图像
        crop_img = crop_imgs[i];
        // 调整图像大小
        this->resize_op_rec.Run(crop_img, resize_img, max_wh_ratio, this->use_tensorrt_);
        
        // 归一化图像
        this->normalize_op_.Run(&resize_img, this->mean_rec, this->scale_rec, this->is_scale_);
        
        // 更新最大调整后的宽高
        max_resize_w = std::max(max_resize_w, resize_img.cols);
        max_resize_h = std::max(max_resize_h, resize_img.rows);
        // 将调整后的图像添加到列表中
        resize_imgs.push_back(resize_img);
    }
    
    // 计算输出缓冲区大小
    int buf_size = 3 * max_resize_h * max_resize_w;
    // 初始化输出结果
    output_rec = std::vector<std::vector<float>>(box_num, std::vector<float>(buf_size, 0.0f));
    
    // 遍历每个文本框
    for (int i = 0; i < box_num; ++i) {
        resize_img_rec = resize_imgs[i];
        // 对调整后的图像进行排列操作
        this->permute_op_.Run(&resize_img_rec, output_rec[i].data());
    }
    
    // 推断输出的形状
    output_shape = {box_num, 3, max_resize_h, max_resize_w};
    // 计算输出形状的元素个数
    out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                              std::multiplies<int>());
    // 计算输出数据缓冲区的大小
    databuf_size_out = out_num * sizeof(float);
    // 分配输出数据缓冲区
    databuf_data_out = MempoolWrapper::instance().malloc(databuf_size_out);
    // 检查分配是否成功
    if (!databuf_data_out) {
      LOG(ERROR) << "Malloc failed, size: " << databuf_size_out;
      return -1;
    }
    // 计算偏移量
    int offset = buf_size * sizeof(float);
    // 将输出数据复制到输出数据缓冲区
    for (int i = 0; i < box_num; ++i) {
      memcpy(databuf_data_out + i * offset, output_rec[i].data(), offset);
    }
    // 将输出数据缓冲区转换为字符型指针
    databuf_char_out = reinterpret_cast<char *>(databuf_data_out);
    // 创建 PaddleBuf 对象
    paddle::PaddleBuf paddleBuf(databuf_char_out, databuf_size_out);
    // 创建 PaddleTensor 对象
    paddle::PaddleTensor tensor_out;
    tensor_out.name = "x";
    tensor_out.dtype = paddle::PaddleDType::FLOAT32;
    tensor_out.shape = output_shape;
    tensor_out.data = paddleBuf;
    // 将输出数据添加到输出向量中
    out->push_back(tensor_out);
  }
  // 移除推断输出数量
  out->erase(out->begin(), out->begin() + infer_outnum);

  // 记录结束时间戳
  int64_t end = timeline.TimeStampUS();
  // 复制输入 Blob 信息到输出 Blob
  CopyBlobInfo(input_blob, output_blob);
  // 添加输出 Blob 的开始时间戳信息
  AddBlobInfo(output_blob, start);
  // 添加输出 Blob 的结束时间戳信息
  AddBlobInfo(output_blob, end);
  // 返回成功
  return 0;
}

// 将 base64 编码的数据转换为 OpenCV 的 Mat 对象
cv::Mat GeneralDetectionOp::Base2Mat(std::string &base64_data) {
  cv::Mat img;
  std::string s_mat;
  // 解码 base64 数据
  s_mat = base64Decode(base64_data.data(), base64_data.size());
  // 将解码后的数据转换为字符向量
  std::vector<char> base64_img(s_mat.begin(), s_mat.end());
  // 使用 OpenCV 解码图像数据
  img = cv::imdecode(base64_img, cv::IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR
  return img;
}

// 解码 base64 编码的数据
std::string GeneralDetectionOp::base64Decode(const char *Data, int DataByte) {
  // base64 解码表
  const char DecodeTable[] = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,
      62, // '+'
      0,  0,  0,
      63,                                     // '/'
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
      0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
      10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
      0,  0,  0,  0,  0,  0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
  };

  std::string strDecode;
  int nValue;
  int i = 0;
  while (i < DataByte) {
    if (*Data != '\r' && *Data != '\n') {
      nValue = DecodeTable[*Data++] << 18;
      nValue += DecodeTable[*Data++] << 12;
      strDecode += (nValue & 0x00FF0000) >> 16;
      if (*Data != '=') {
        nValue += DecodeTable[*Data++] << 6;
        strDecode += (nValue & 0x0000FF00) >> 8;
        if (*Data != '=') {
          nValue += DecodeTable[*Data++];
          strDecode += nValue & 0x000000FF;
        }
      }
      i += 4;
    } else // 回车换行,跳过
    {
      Data++;
      i++;
    }
  }
  return strDecode;
}

cv::Mat
// 获取旋转裁剪后的图像
cv::Mat GeneralDetectionOp::GetRotateCropImage(const cv::Mat &srcimage,
                                               std::vector<std::vector<int>> box) {
  // 复制输入图像
  cv::Mat image;
  srcimage.copyTo(image);
  // 复制边界框坐标
  std::vector<std::vector<int>> points = box;

  // 提取边界框 x 和 y 坐标
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  // 计算裁剪区域的左、右、上、下边界
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  // 裁剪图像
  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  // 更新边界框坐标
  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  // 计算裁剪图像的宽度和高度
  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));

  // 定义标准四边形的四个顶点
  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  // 定义边界框四个顶点
  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  // 获取透视变换矩阵
  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  // 进行透视变换
  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  // 如果变换后的图像高度大于宽度的1.5倍，则进行矩阵转置
  if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
  }
}
    # 沿水平轴翻转图像
    cv::flip(srcCopy, srcCopy, 0);
    # 返回翻转后的图像
    return srcCopy;
  } else {
    # 如果条件不满足，返回原始图像
    return dst_img;
  }
}

// 定义通用检测操作
DEFINE_OP(GeneralDetectionOp);

// 结束 serving 命名空间
} // namespace serving

// 结束 paddle_serving 命名空间
} // namespace paddle_serving

// 结束 baidu 命名空间
} // namespace baidu
```