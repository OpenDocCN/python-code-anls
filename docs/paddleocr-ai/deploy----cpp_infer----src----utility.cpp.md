# `.\PaddleOCR\deploy\cpp_infer\src\utility.cpp`

```
// 包含必要的头文件
#include <dirent.h>
#include <include/utility.h>
#include <iostream>
#include <ostream>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// 命名空间定义
namespace PaddleOCR {

// 从文件中读取字典内容并返回
std::vector<std::string> Utility::ReadDict(const std::string &path) {
  // 打开文件流
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  // 如果文件流有效
  if (in) {
    // 逐行读取文件内容并存储到向量中
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    // 如果文件不存在，则输出错误信息并退出程序
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  // 返回存储文件内容的向量
  return m_vec;
}

// 可视化检测框
void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                              const std::vector<OCRPredictResult> &ocr_result,
                              const std::string &save_path) {
  // 复制原始图像
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  // 遍历OCR结果
  for (int n = 0; n < ocr_result.size(); n++) {
    cv::Point rook_points[4];
    // 绘制检测框
    for (int m = 0; m < ocr_result[n].box.size(); m++) {
      rook_points[m] =
          cv::Point(int(ocr_result[n].box[m][0]), int(ocr_result[n].box[m][1]));
    }

    const cv::Point *ppt[1] = {rook_points};
    int npt[] = {4};
    // 绘制多边形
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  // 将可视化结果保存为图像文件
  cv::imwrite(save_path, img_vis);
  std::cout << "The detection visualized image saved in " + save_path
            << std::endl;
}
}
// 可视化边界框并保存结果图像
void Utility::VisualizeBboxes(const cv::Mat &srcimg,
                              const StructurePredictResult &structure_result,
                              const std::string &save_path) {
  // 复制原始图像
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  // 裁剪图像
  img_vis = crop_image(img_vis, structure_result.box);
  // 遍历每个单元格的边界框
  for (int n = 0; n < structure_result.cell_box.size(); n++) {
    // 如果边界框有8个点，则绘制多边形
    if (structure_result.cell_box[n].size() == 8) {
      cv::Point rook_points[4];
      // 提取每个点的坐标
      for (int m = 0; m < structure_result.cell_box[n].size(); m += 2) {
        rook_points[m / 2] =
            cv::Point(int(structure_result.cell_box[n][m]),
                      int(structure_result.cell_box[n][m + 1]));
      }
      const cv::Point *ppt[1] = {rook_points};
      int npt[] = {4};
      // 绘制多边形
      cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    } 
    // 如果边界框有4个点，则绘制矩形
    else if (structure_result.cell_box[n].size() == 4) {
      cv::Point rook_points[2];
      // 提取矩形的两个点坐标
      rook_points[0] = cv::Point(int(structure_result.cell_box[n][0]),
                                 int(structure_result.cell_box[n][1]));
      rook_points[1] = cv::Point(int(structure_result.cell_box[n][2]),
                                 int(structure_result.cell_box[n][3]));
      // 绘制矩形
      cv::rectangle(img_vis, rook_points[0], rook_points[1], CV_RGB(0, 255, 0),
                    2, 8, 0);
    }
  }

  // 保存可视化结果图像
  cv::imwrite(save_path, img_vis);
  std::cout << "The table visualized image saved in " + save_path << std::endl;
}

// 获取指定目录下的所有文件
void Utility::GetAllFiles(const char *dir_name,
                          std::vector<std::string> &all_inputs) {
  // 检查目录名是否为空
  if (NULL == dir_name) {
    std::cout << " dir_name is null ! " << std::endl;
    return;
  }
  struct stat s;
  // 获取目录信息
  stat(dir_name, &s);
  // 如果不是目录，则将其添加到文件列表中
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    all_inputs.push_back(dir_name);
    return;
  } else {
    struct dirent *filename; // readdir() 的返回值
    DIR *dir;                // opendir() 的返回值
    // 打开指定目录，返回目录指针
    dir = opendir(dir_name);
    // 如果打开目录失败，输出错误信息并返回
    if (NULL == dir) {
      std::cout << "Can not open dir " << dir_name << std::endl;
      return;
    }
    // 打开目录成功的提示信息
    std::cout << "Successfully opened the dir !" << std::endl;
    // 遍历目录中的文件
    while ((filename = readdir(dir)) != NULL) {
      // 如果文件名为"."或"..", 则跳过
      if (strcmp(filename->d_name, ".") == 0 ||
          strcmp(filename->d_name, "..") == 0)
        continue;
      // 将目录名和文件名拼接成完整路径，添加到all_inputs中
      all_inputs.push_back(dir_name + std::string("/") +
                           std::string(filename->d_name));
    }
  }
// 获取旋转裁剪后的图像
cv::Mat Utility::GetRotateCropImage(const cv::Mat &srcimage,
                                    std::vector<std::vector<int>> box) {
  // 复制输入图像
  cv::Mat image;
  srcimage.copyTo(image);
  // 获取边界框的四个顶点坐标
  std::vector<std::vector<int>> points = box;

  // 提取边界框的 x 和 y 坐标
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  // 计算裁剪区域的左上角和右下角坐标
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  // 裁剪图像
  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  // 更新顶点坐标
  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  // 计算裁剪后图像的宽度和高度
  int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                pow(points[0][1] - points[1][1], 2)));
  int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                 pow(points[0][1] - points[3][1], 2)));

  // 定义标准四个顶点
  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  // 转换顶点坐标为浮点型
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

  // 如果高度大于宽度的1.5倍，则进行矩阵转置
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

// 对数组进行排序并返回排序后的索引数组
std::vector<int> Utility::argsort(const std::vector<float> &array) {
  // 获取数组长度
  const int array_len(array.size());
  // 创建与数组长度相同的索引数组
  std::vector<int> array_index(array_len, 0);
  // 初始化索引数组
  for (int i = 0; i < array_len; ++i)
    array_index[i] = i;

  // 对索引数组根据数组元素大小进行排序
  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

// 获取文件路径中的文件名
std::string Utility::basename(const std::string &filename) {
  if (filename.empty()) {
    return "";
  }

  auto len = filename.length();
  auto index = filename.find_last_of("/\\");

  if (index == std::string::npos) {
    return filename;
  }

  if (index + 1 >= len) {

    len--;
    index = filename.substr(0, len).find_last_of("/\\");

    if (len == 0) {
      return filename;
    }

    if (index == 0) {
      return filename.substr(1, len - 1);
    }

    if (index == std::string::npos) {
      return filename.substr(0, len);
    }

    return filename.substr(index + 1, len - index - 1);
  }

  return filename.substr(index + 1, len - index);
}

// 检查路径是否存在
bool Utility::PathExists(const std::string &path) {
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
#endif // !_WIN32
}

// 创建目录
void Utility::CreateDir(const std::string &path) {
#ifdef _WIN32
  _mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0777);
#endif // !_WIN32
}

// 打印 OCR 预测结果
void Utility::print_result(const std::vector<OCRPredictResult> &ocr_result) {
  for (int i = 0; i < ocr_result.size(); i++) {
    std::cout << i << "\t";
    // det
    std::vector<std::vector<int>> boxes = ocr_result[i].box;
    if (boxes.size() > 0) {
      std::cout << "det boxes: [";
      for (int n = 0; n < boxes.size(); n++) {
        std::cout << '[' << boxes[n][0] << ',' << boxes[n][1] << "]";
        if (n != boxes.size() - 1) {
          std::cout << ',';
        }
      }
      std::cout << "] ";
    }
    // rec
    // 如果 OCR 结果中的置信度不为 -1.0
    if (ocr_result[i].score != -1.0) {
      // 输出识别文本和置信度
      std::cout << "rec text: " << ocr_result[i].text
                << " rec score: " << ocr_result[i].score << " ";
    }

    // cls
    // 如果 OCR 结果中的类别标签不为 -1
    if (ocr_result[i].cls_label != -1) {
      // 输出类别标签和类别置信度
      std::cout << "cls label: " << ocr_result[i].cls_label
                << " cls score: " << ocr_result[i].cls_score;
    }
    // 输出换行符
    std::cout << std::endl;
  }
// 裁剪图像，根据给定的边界框坐标
cv::Mat Utility::crop_image(cv::Mat &img, const std::vector<int> &box) {
  // 初始化裁剪后的图像
  cv::Mat crop_im;
  // 计算裁剪区域的左上角坐标
  int crop_x1 = std::max(0, box[0]);
  int crop_y1 = std::max(0, box[1]);
  // 计算裁剪区域的右下角坐标
  int crop_x2 = std::min(img.cols - 1, box[2] - 1);
  int crop_y2 = std::min(img.rows - 1, box[3] - 1);

  // 创建裁剪后的图像
  crop_im = cv::Mat::zeros(box[3] - box[1], box[2] - box[0], 16);
  // 获取裁剪区域的窗口
  cv::Mat crop_im_window =
      crop_im(cv::Range(crop_y1 - box[1], crop_y2 + 1 - box[1]),
              cv::Range(crop_x1 - box[0], crop_x2 + 1 - box[0]));
  // 获取原图像中对应的区域
  cv::Mat roi_img =
      img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));
  // 将原图像区域复制到裁剪图像中
  crop_im_window += roi_img;
  // 返回裁剪后的图像
  return crop_im;
}

// 将浮点数类型的边界框坐标转换为整数类型，并调用上面的裁剪函数
cv::Mat Utility::crop_image(cv::Mat &img, const std::vector<float> &box) {
  // 将浮点数类型的边界框坐标转换为整数类型
  std::vector<int> box_int = {(int)box[0], (int)box[1], (int)box[2],
                              (int)box[3]};
  // 调用裁剪函数
  return crop_image(img, box_int);
}

// 对 OCR 预测结果中的边界框进行排序
void Utility::sorted_boxes(std::vector<OCRPredictResult> &ocr_result) {
  // 按照指定的比较函数对边界框进行排序
  std::sort(ocr_result.begin(), ocr_result.end(), Utility::comparison_box);
  // 如果结果集不为空，则进行进一步处理
  if (ocr_result.size() > 0) {
    // 遍历结果集，对相邻的边界框进行比较和交换
    for (int i = 0; i < ocr_result.size() - 1; i++) {
      for (int j = i; j >= 0; j--) {
        // 如果相邻边界框的垂直距离小于 10 且水平位置不正确，则交换它们
        if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
            (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0])) {
          std::swap(ocr_result[i], ocr_result[i + 1]);
        }
      }
    }
  }
}

// 将包含四个点坐标的边界框转换为左上角和右下角坐标形式
std::vector<int> Utility::xyxyxyxy2xyxy(std::vector<std::vector<int>> &box) {
  // 收集 x 和 y 坐标
  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  // 计算左上角和右下角坐标
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  // 组装成左上角和右下角坐标形式的边界框
  std::vector<int> box1(4, 0);
  box1[0] = left;
  box1[1] = top;
  box1[2] = right;
  box1[3] = bottom;
  // 返回转换后的边界框
  return box1;
}
// 将输入的包含8个元素的box向量转换为包含4个元素的box1向量
std::vector<int> Utility::xyxyxyxy2xyxy(std::vector<int> &box) {
  // 从box中提取x坐标，存储在x_collect数组中
  int x_collect[4] = {box[0], box[2], box[4], box[6]};
  // 从box中提取y坐标，存储在y_collect数组中
  int y_collect[4] = {box[1], box[3], box[5], box[7]};
  // 计算x坐标的最小值作为left
  int left = int(*std::min_element(x_collect, x_collect + 4));
  // 计算x坐标的最大值作为right
  int right = int(*std::max_element(x_collect, x_collect + 4));
  // 计算y坐标的最小值作为top
  int top = int(*std::min_element(y_collect, y_collect + 4));
  // 计算y坐标的最大值作为bottom
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  // 创建包含left、top、right、bottom的box1向量并返回
  std::vector<int> box1(4, 0);
  box1[0] = left;
  box1[1] = top;
  box1[2] = right;
  box1[3] = bottom;
  return box1;
}

// 快速计算e的x次方
float Utility::fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  // 使用浮点数的位操作快速计算e的x次方
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

// 实现softmax激活函数
std::vector<float>
Utility::activation_function_softmax(std::vector<float> &src) {
  int length = src.size();
  std::vector<float> dst;
  dst.resize(length);
  // 计算src中的最大值作为alpha
  const float alpha = float(*std::max_element(&src[0], &src[0 + length]));
  float denominator{0};

  // 计算softmax函数的分子和分母
  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  // 归一化处理得到softmax结果
  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return dst;
}

// 计算两个box的交并比
float Utility::iou(std::vector<int> &box1, std::vector<int> &box2) {
  // 计算box1和box2的面积
  int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
  int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

  // 计算两个box的交集面积和并集面积
  int sum_area = area1 + area2;
  int x1 = std::max(box1[0], box2[0]);
  int y1 = std::max(box1[1], box2[1]);
  int x2 = std::min(box1[2], box2[2]);
  int y2 = std::min(box1[3], box2[3]);

  // 判断是否有交集，计算交并比
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    int intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}
// 计算两个框的交并比
float Utility::iou(std::vector<float> &box1, std::vector<float> &box2) {
  // 计算第一个框的面积
  float area1 = std::max((float)0.0, box1[2] - box1[0]) *
                std::max((float)0.0, box1[3] - box1[1]);
  // 计算第二个框的面积
  float area2 = std::max((float)0.0, box2[2] - box2[0]) *
                std::max((float)0.0, box2[3] - box2[1]);

  // 计算两个框的总面积
  float sum_area = area1 + area2;

  // 计算交集矩形的左上角和右下角坐标
  float x1 = std::max(box1[0], box2[0]);
  float y1 = std::max(box1[1], box2[1]);
  float x2 = std::min(box1[2], box2[2]);
  float y2 = std::min(box1[3], box2[3]);

  // 判断是否存在交集
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    // 计算交集的面积
    float intersect = (x2 - x1) * (y2 - y1);
    // 计算交并比，并避免除零错误
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

} // namespace PaddleOCR
```