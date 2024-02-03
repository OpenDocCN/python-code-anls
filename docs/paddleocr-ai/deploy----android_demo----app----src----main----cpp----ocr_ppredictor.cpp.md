# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_ppredictor.cpp`

```
// 包含头文件
#include "ocr_ppredictor.h"
#include "common.h"
#include "ocr_cls_process.h"
#include "ocr_crnn_process.h"
#include "ocr_db_post_process.h"
#include "preprocess.h"

// 命名空间
namespace ppredictor {

// 构造函数，初始化 OCR_PPredictor 对象
OCR_PPredictor::OCR_PPredictor(const OCR_Config &config) : _config(config) {}

// 初始化函数，加载模型内容
int OCR_PPredictor::init(const std::string &det_model_content,
                         const std::string &rec_model_content,
                         const std::string &cls_model_content) {
  // 初始化检测模型预测器
  _det_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl,_config.thread_num, NET_OCR, _config.mode});
  _det_predictor->init_nb(det_model_content);

  // 初始化识别模型预测器
  _rec_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl,_config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _rec_predictor->init_nb(rec_model_content);

  // 初始化分类模型预测器
  _cls_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl,_config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _cls_predictor->init_nb(cls_model_content);
  return RETURN_OK;
}

// 从文件初始化模型
int OCR_PPredictor::init_from_file(const std::string &det_model_path,
                                   const std::string &rec_model_path,
                                   const std::string &cls_model_path) {
  // 从文件初始化检测模型预测器
  _det_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl, _config.thread_num, NET_OCR, _config.mode});
  _det_predictor->init_from_file(det_model_path);

  // 从文件初始化识别模型预测器
  _rec_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl,_config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _rec_predictor->init_from_file(rec_model_path);

  // 从文件初始化分类模型预测器
  _cls_predictor = std::unique_ptr<PPredictor>(
      new PPredictor{_config.use_opencl,_config.thread_num, NET_OCR_INTERNAL, _config.mode});
  _cls_predictor->init_from_file(cls_model_path);
  return RETURN_OK;
}

/**
 * for debug use, show result of First Step
 * @param filter_boxes
 * @param boxes
 * @param srcimg
 */
static void
visual_img(const std::vector<std::vector<std::vector<int>>> &filter_boxes,
           const std::vector<std::vector<std::vector<int>>> &boxes,
           const cv::Mat &srcimg) {
  // 可视化函数，用于绘制图像
  cv::Point rook_points[filter_boxes.size()][4];
  // 遍历 filter_boxes，将其转换为 cv::Point 类型的数组
  for (int n = 0; n < filter_boxes.size(); n++) {
    for (int m = 0; m < filter_boxes[0].size(); m++) {
      rook_points[n][m] =
          cv::Point(int(filter_boxes[n][m][0]), int(filter_boxes[n][m][1]));
    }
  }

  cv::Mat img_vis;
  // 复制原始图像到 img_vis
  srcimg.copyTo(img_vis);
  // 遍历 boxes，绘制多边形
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }
  // 保存可视化结果图像到指定路径
  cv::imwrite("/sdcard/1/vis.png", img_vis);
}

std::vector<OCRPredictResult>
OCR_PPredictor::infer_ocr(cv::Mat &origin,int max_size_len, int run_det, int run_cls, int run_rec) {
  LOGI("ocr cpp start *****************");
  LOGI("ocr cpp det: %d, cls: %d, rec: %d", run_det, run_cls, run_rec);
  std::vector<OCRPredictResult> ocr_results;
  // 如果需要运行检测模型
  if(run_det){
    infer_det(origin, max_size_len, ocr_results);
  }
  // 如果需要运行识别模型
  if(run_rec){
    // 如果没有检测结果，则创建一个空的 OCRPredictResult 对象
    if(ocr_results.size()==0){
      OCRPredictResult res;
      ocr_results.emplace_back(std::move(res));
    }
    // 遍历检测结果，运行识别模型
    for(int i = 0; i < ocr_results.size();i++) {
      infer_rec(origin, run_cls, ocr_results[i]);
    }
  }else if(run_cls){
    // 运行分类模型，将结果存储到 OCRPredictResult 中
    ClsPredictResult cls_res = infer_cls(origin);
    OCRPredictResult res;
    res.cls_score = cls_res.cls_score;
    res.cls_label = cls_res.cls_label;
    ocr_results.push_back(res);
  }

  LOGI("ocr cpp end *****************");
  return ocr_results;
}

cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  // 计算缩放比例
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
  } else {
    // 如果宽度大于最大尺寸长度，则计算缩放比例
    ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
  }
}

// 计算缩放后的高度和宽度
int resize_h = static_cast<int>(float(h) * ratio);
int resize_w = static_cast<int>(float(w) * ratio);

// 调整高度为32的倍数
if (resize_h % 32 == 0)
  resize_h = resize_h;
else if (resize_h / 32 < 1 + 1e-5)
  resize_h = 32;
else
  resize_h = (resize_h / 32 - 1) * 32;

// 调整宽度为32的倍数
if (resize_w % 32 == 0)
  resize_w = resize_w;
else if (resize_w / 32 < 1 + 1e-5)
  resize_w = 32;
else
  resize_w = (resize_w / 32 - 1) * 32;

// 调整图像大小
cv::Mat resize_img;
cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

// 计算高度和宽度的缩放比例并存入ratio_hw
ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));

// 返回调整大小后的图像
return resize_img;
// 定义 OCR_PPredictor 类的 infer_det 方法，用于进行目标检测推理
void OCR_PPredictor::infer_det(cv::Mat &origin, int max_size_len, std::vector<OCRPredictResult> &ocr_results) {
  // 定义均值和缩放比例
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};

  // 获取目标检测模型的输入
  PredictorInput input = _det_predictor->get_first_input();

  // 定义变量 ratio_hw 和 input_image，调用 DetResizeImg 函数进行图像尺寸调整
  std::vector<float> ratio_hw;
  cv::Mat input_image = DetResizeImg(origin, max_size_len, ratio_hw);
  // 将 input_image 转换为 CV_32FC3 格式，并进行归一化
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  // 获取 input_image 的数据指针
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  // 计算 input_size
  int input_size = input_image.rows * input_image.cols;

  // 设置输入的维度
  input.set_dims({1, 3, input_image.rows, input_image.cols});

  // 对输入数据进行均值和缩放处理
  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean, scale);
  // 打印输入图像的形状信息
  LOGI("ocr cpp det shape %d,%d", input_image.rows,input_image.cols);
  // 进行目标检测推理，获取结果
  std::vector<PredictorOutput> results = _det_predictor->infer();
  // 获取推理结果的第一个输出
  PredictorOutput &res = results.at(0);
  // 计算过滤后的边界框
  std::vector<std::vector<std::vector<int>>> filtered_box = calc_filtered_boxes(
          res.get_float_data(), res.get_size(), input_image.rows, input_image.cols, origin);
  // 打印过滤后的边界框的大小
  LOGI("ocr cpp det Filter_box size %ld", filtered_box.size());

  // 遍历过滤后的边界框，将结果保存到 ocr_results 中
  for(int i = 0;i<filtered_box.size();i++){
    LOGI("ocr cpp box  %d,%d,%d,%d,%d,%d,%d,%d", filtered_box[i][0][0],filtered_box[i][0][1], filtered_box[i][1][0],filtered_box[i][1][1], filtered_box[i][2][0],filtered_box[i][2][1], filtered_box[i][3][0],filtered_box[i][3][1]);
    OCRPredictResult res;
    res.points = filtered_box[i];
    ocr_results.push_back(res);
  }
}

// 定义 OCR_PPredictor 类的 infer_rec 方法，用于进行文本识别推理
void OCR_PPredictor::infer_rec(const cv::Mat &origin_img, int run_cls, OCRPredictResult& ocr_result) {
  // 定义均值和缩放比例
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  // 定义维度
  std::vector<int64_t> dims = {1, 3, 0, 0};

  // 获取文本识别模型的输入
  PredictorInput input = _rec_predictor->get_first_input();

  // 获取 OCR 结果中的边界框信息
  const std::vector<std::vector<int>> &box = ocr_result.points;
  cv::Mat crop_img;
  // 如果边界框数量大于 0
  // 如果需要裁剪图像
  crop_img = get_rotate_crop_image(origin_img, box);
  // 如果不需要裁剪图像
  else{
    crop_img = origin_img;
  }

  // 如果需要运行分类器
  if(run_cls){
    // 对裁剪后的图像进行分类推断
    ClsPredictResult cls_res = infer_cls(crop_img);
    crop_img = cls_res.img;
    ocr_result.cls_score = cls_res.cls_score;
    ocr_result.cls_label = cls_res.cls_label;
  }

  // 计算裁剪后图像的宽高比
  float wh_ratio = float(crop_img.cols) / float(crop_img.rows);
  // 调整图像大小以适应 CRNN 模型输入
  cv::Mat input_image = crnn_resize_img(crop_img, wh_ratio);
  // 将图像数据转换为浮点型并归一化
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  // 获取输入图像数据指针
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  // 计算输入图像大小
  int input_size = input_image.rows * input_image.cols;

  // 设置输入数据维度
  dims[2] = input_image.rows;
  dims[3] = input_image.cols;
  input.set_dims(dims);

  // 对输入数据进行均值和标准差归一化处理
  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean,
                  scale);

  // 进行 CRNN 模型推断
  std::vector<PredictorOutput> results = _rec_predictor->infer();
  // 获取推断结果数据
  const float *predict_batch = results.at(0).get_float_data();
  const std::vector<int64_t> predict_shape = results.at(0).get_shape();

  // CTC 解码
  int argmax_idx;
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  float max_value = 0.0f;

  // 遍历预测结果进行解码
  for (int n = 0; n < predict_shape[1]; n++) {
    argmax_idx = int(argmax(&predict_batch[n * predict_shape[2]],
                            &predict_batch[(n + 1) * predict_shape[2]));
    max_value =
        float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                &predict_batch[(n + 1) * predict_shape[2]]));
    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      ocr_result.word_index.push_back(argmax_idx);
    }
    last_index = argmax_idx;
  }
  score /= count;
  ocr_result.score = score;
  // 打印识别结果的单词数量
  LOGI("ocr cpp rec word size %ld", count);
// 定义 OCR_PPredictor 类的 infer_cls 方法，用于进行分类推断
ClsPredictResult OCR_PPredictor::infer_cls(const cv::Mat &img, float thresh) {
  // 定义均值向量
  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  // 定义缩放比例向量
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  // 定义维度向量
  std::vector<int64_t> dims = {1, 3, 0, 0};

  // 获取分类器输入
  PredictorInput input = _cls_predictor->get_first_input();

  // 调整输入图像大小
  cv::Mat input_image = cls_resize_img(img);
  // 将输入图像转换为浮点型，范围为[0, 1]
  input_image.convertTo(input_image, CV_32FC3, 1 / 255.0f);
  // 获取输入图像数据指针
  const float *dimg = reinterpret_cast<const float *>(input_image.data);
  // 计算输入图像大小
  int input_size = input_image.rows * input_image.cols;

  // 更新维度信息
  dims[2] = input_image.rows;
  dims[3] = input_image.cols;
  input.set_dims(dims);

  // 对输入图像进行均值和缩放处理
  neon_mean_scale(dimg, input.get_mutable_float_data(), input_size, mean,
                  scale);

  // 进行推断
  std::vector<PredictorOutput> results = _cls_predictor->infer();

  // 获取分类结果分数
  const float *scores = results.at(0).get_float_data();
  float score = 0;
  int label = 0;
  // 遍历分类结果，找到最高分数的类别
  for (int64_t i = 0; i < results.at(0).get_size(); i++) {
    LOGI("ocr cpp cls output scores [%f]", scores[i]);
    if (scores[i] > score) {
      score = scores[i];
      label = i;
    }
  }
  // 复制原始图像
  cv::Mat srcimg;
  img.copyTo(srcimg);
  // 如果类别为奇数且分数高于阈值，则旋转图像
  if (label % 2 == 1 && score > thresh) {
    cv::rotate(srcimg, srcimg, 1);
  }
  // 创建分类结果对象
  ClsPredictResult res;
  res.cls_label = label;
  res.cls_score = score;
  res.img = srcimg;
  LOGI("ocr cpp cls word cls %ld, %f", label, score);
  // 返回分类结果
  return res;
}

// 定义一个返回三维整数向量的函数
std::vector<std::vector<std::vector<int>>>
// 计算过滤后的框，根据预测值、输出尺寸和原始图像
std::vector<std::vector<std::vector<int>>> OCR_PPredictor::calc_filtered_boxes(const float *pred, int pred_size,
                                    int output_height, int output_width,
                                    const cv::Mat &origin) {
  // 设置阈值和最大值
  const double threshold = 0.3;
  const double maxvalue = 1;

  // 创建一个与输出尺寸相同的零矩阵
  cv::Mat pred_map = cv::Mat::zeros(output_height, output_width, CV_32F);
  // 将预测值复制到矩阵中
  memcpy(pred_map.data, pred, pred_size * sizeof(float));
  // 将预测值矩阵转换为 CV_8UC1 类型
  cv::Mat cbuf_map;
  pred_map.convertTo(cbuf_map, CV_8UC1);

  // 创建一个二值化的矩阵
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);

  // 从位图中获取框
  std::vector<std::vector<std::vector<int>>> boxes =
      boxes_from_bitmap(pred_map, bit_map);
  // 计算高度和宽度的比例
  float ratio_h = output_height * 1.0f / origin.rows;
  float ratio_w = output_width * 1.0f / origin.cols;
  // 过滤检测结果的框
  std::vector<std::vector<std::vector<int>>> filter_boxes =
      filter_tag_det_res(boxes, ratio_h, ratio_w, origin);
  // 返回过滤后的框
  return filter_boxes;
}

// 后处理识别结果的单词索引
std::vector<int> OCR_PPredictor::postprocess_rec_word_index(const PredictorOutput &res) {
  // 获取整型数据和长度信息
  const int *rec_idx = res.get_int_data();
  const std::vector<std::vector<uint64_t>> rec_idx_lod = res.get_lod();

  std::vector<int> pred_idx;
  // 遍历索引数据，获取单词索引
  for (int n = int(rec_idx_lod[0][0]); n < int(rec_idx_lod[0][1] * 2); n += 2) {
    pred_idx.emplace_back(rec_idx[n]);
  }
  // 返回单词索引
  return pred_idx;
}

// 后处理识别结果的分数
float OCR_PPredictor::postprocess_rec_score(const PredictorOutput &res) {
  // 获取浮点数据、形状和长度信息
  const float *predict_batch = res.get_float_data();
  const std::vector<int64_t> predict_shape = res.get_shape();
  const std::vector<std::vector<uint64_t>> predict_lod = res.get_lod();
  int blank = predict_shape[1];
  float score = 0.f;
  int count = 0;
  // 遍历预测数据，计算分数
  for (int n = predict_lod[0][0]; n < predict_lod[0][1] - 1; n++) {
    int argmax_idx = argmax(predict_batch + n * predict_shape[1],
                            predict_batch + (n + 1) * predict_shape[1]);
    float max_value = predict_batch[n * predict_shape[1] + argmax_idx];
    // 如果不是空白字符，则累加分数
    if (blank - 1 - argmax_idx > 1e-5) {
      score += max_value;
      count += 1;
    }
  }
  // 如果计数为0，则记录错误日志
  if (count == 0) {
    LOGE("calc score count 0");
  } else {
    // 否则计算平均分数
    score /= count;
  }
  // 记录信息日志，显示计算得到的分数
  LOGI("calc score: %f", score);
  // 返回计算得到的分数
  return score;
# 结束 OCR_PPredictor 类的定义
}

# 返回 OCR_PPredictor 类的网络类型标志 NET_OCR
NET_TYPE OCR_PPredictor::get_net_flag() const { return NET_OCR; }
# 结束 get_net_flag 方法的定义
}
```