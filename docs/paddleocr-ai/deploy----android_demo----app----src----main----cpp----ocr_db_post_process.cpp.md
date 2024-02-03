# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\ocr_db_post_process.cpp`

```
// 包含版权声明和许可信息
// 引入所需的头文件
#include "ocr_clipper.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <vector>

// 计算轮廓面积和距离
static void getcontourarea(float **box, float unclip_ratio, float &distance) {
  // 定义点的数量和初始面积、距离
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  // 计算面积和距离
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  // 计算最终面积
  area = fabs(float(area / 2.0));

  // 计算距离
  distance = area * unclip_ratio / dist;
}

// 对文本框进行解除裁剪
static cv::RotatedRect unclip(float **box) {
  // 设置解除裁剪比例和初始距离
  float unclip_ratio = 2.0;
  float distance = 1.0;

  // 调用函数计算面积和距离
  getcontourarea(box, unclip_ratio, distance);

  // 创建 ClipperOffset 对象
  ClipperLib::ClipperOffset offset;
  // 创建路径对象并添加点
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
    << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
    << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
    // 创建一个 ClipperLib::IntPoint 对象，使用 box[3][0] 和 box[3][1] 的整数部分作为坐标
    ClipperLib::IntPoint p(int(box[3][0]), int(box[3][1]));
    // 将路径 p 添加到 offset 中，采用 ClipperLib::jtRound 舍入方式和 ClipperLib::etClosedPolygon 封闭多边形方式
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    // 创建一个 ClipperLib::Paths 对象 soln
    ClipperLib::Paths soln;
    // 执行 offset 操作，将结果存储在 soln 中，距离为 distance
    offset.Execute(soln, distance);
    // 创建一个 std::vector<cv::Point2f> 对象 points

    // 遍历 soln 中的路径
    for (int j = 0; j < soln.size(); j++) {
        // 遍历当前路径中的点
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            // 将当前点的 X 和 Y 坐标作为 cv::Point2f 对象的坐标，添加到 points 中
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    // 使用 points 中的点创建一个最小外接矩形，存储在 res 中
    cv::RotatedRect res = cv::minAreaRect(points);

    // 返回最小外接矩形 res
    return res;
static float **Mat2Vec(cv::Mat mat) {
  // 创建一个二维浮点数数组，用于存储矩阵数据
  auto **array = new float *[mat.rows];
  // 遍历矩阵的行数，为每一行创建一个浮点数数组
  for (int i = 0; i < mat.rows; ++i) {
    array[i] = new float[mat.cols];
  }
  // 将矩阵数据复制到二维数组中
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

static void quickSort(float **s, int l, int r) {
  // 使用快速排序算法对二维数组进行排序
  if (l < r) {
    int i = l, j = r;
    float x = s[l][0];
    float *xp = s[l];
    while (i < j) {
      while (i < j && s[j][0] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(s[i++], s[j]);
      }
      while (i < j && s[i][0] < x) {
        i++;
      }
      if (i < j) {
        std::swap(s[j--], s[i]);
      }
    }
    s[i] = xp;
    quickSort(s, l, i - 1);
    quickSort(s, i + 1, r);
  }
}

static void quickSort_vector(std::vector<std::vector<int>> &box, int l, int r,
                             int axis) {
  // 使用快速排序算法对二维向量进行排序
  if (l < r) {
    int i = l, j = r;
    int x = box[l][axis];
    std::vector<int> xp(box[l]);
    while (i < j) {
      while (i < j && box[j][axis] >= x) {
        j--;
      }
      if (i < j) {
        std::swap(box[i++], box[j]);
      }
      while (i < j && box[i][axis] < x) {
        i++;
      }
      if (i < j) {
        std::swap(box[j--], box[i]);
      }
    }
    box[i] = xp;
    quickSort_vector(box, l, i - 1, axis);
    quickSort_vector(box, i + 1, r, axis);
  }
}

static std::vector<std::vector<int>>
order_points_clockwise(std::vector<std::vector<int>> pts) {
  // 复制输入的二维向量
  std::vector<std::vector<int>> box = pts;
  // 对二维向量按指定轴进行快速排序
  quickSort_vector(box, 0, int(box.size() - 1), 0);
  // 分别取左上角和右下角的两个点
  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  // 如果左上角的点的 y 坐标大于右上角的点的 y 坐标，则交换两个点
  if (leftmost[0][1] > leftmost[1][1]) {
    std::swap(leftmost[0], leftmost[1]);
  }

  // 如果右上角的点的 y 坐标大于右下角的点的 y 坐标，则交换两个点
  if (rightmost[0][1] > rightmost[1][1]) {
    // 交换 rightmost 数组中的两个元素
    std::swap(rightmost[0], rightmost[1]);
  }

  // 创建一个二维向量 rect，包含 leftmost[0], rightmost[0], rightmost[1], leftmost[1] 四个元素
  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  // 返回 rect 向量
  return rect;
// 获取旋转矩形的最小边界框的四个顶点坐标，并计算最小边长，存储在ssid中
static float **get_mini_boxes(cv::RotatedRect box, float &ssid) {
  ssid = box.size.width >= box.size.height ? box.size.height : box.size.width;

  // 获取旋转矩形的四个顶点坐标
  cv::Mat points;
  cv::boxPoints(box, points);
  // 对顶点坐标进行排序
  auto array = Mat2Vec(points);
  quickSort(array, 0, 3);

  // 根据顶点坐标的y值大小关系，重新排序顶点坐标
  float *idx1 = array[0], *idx2 = array[1], *idx3 = array[2], *idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  // 更新顶点坐标数组
  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

// 对输入值x进行范围限制，确保在[min, max]之间
template <class T> T clamp(T x, T min, T max) {
  if (x > max) {
    return max;
  }
  if (x < min) {
    return min;
  }
  return x;
}

// 对输入值x进行范围限制，确保在[min, max]之间
static float clampf(float x, float min, float max) {
  if (x > max)
    return max;
  if (x < min)
    return min;
  return x;
}
# 计算包围框的得分
float box_score_fast(float **box_array, cv::Mat pred) {
  # 将二维数组赋值给指针数组
  auto array = box_array;
  # 获取预测图像的宽度和高度
  int width = pred.cols;
  int height = pred.rows;

  # 提取包围框的 x 坐标
  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  # 提取包围框的 y 坐标
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  # 计算包围框的 xmin 和 xmax
  int xmin = clamp(int(std::floorf(*(std::min_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceilf(*(std::max_element(box_x, box_x + 4)))), 0,
                   width - 1);
  # 计算包围框的 ymin 和 ymax
  int ymin = clamp(int(std::floorf(*(std::min_element(box_y, box_y + 4)))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceilf(*(std::max_element(box_y, box_y + 4)))), 0,
                   height - 1);

  # 创建一个与包围框大小相同的掩码图像
  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  # 计算包围框的四个顶点相对于左上角的偏移
  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  # 在掩码图像上填充多边形区域
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  # 从预测图像中裁剪出包围框区域
  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  # 计算裁剪图像在掩码区域内的平均值作为得分
  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

# 未完整的代码，缺少后续部分
std::vector<std::vector<std::vector<int>>>
// 从预测矩阵和位图中获取边界框
std::vector<std::vector<std::vector<int>>> boxes_from_bitmap(const cv::Mat &pred, const cv::Mat &bitmap) {
    // 定义最小尺寸、最大候选框数量和边界框阈值
    const int min_size = 3;
    const int max_candidates = 1000;
    const float box_thresh = 0.5;

    // 获取位图的宽度和高度
    int width = bitmap.cols;
    int height = bitmap.rows;

    // 定义轮廓和层次结构
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 查找位图中的轮廓
    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 确定要处理的轮廓数量
    int num_contours = contours.size() >= max_candidates ? max_candidates : contours.size();

    // 存储边界框的三维向量
    std::vector<std::vector<std::vector<int>>> boxes;

    // 遍历每个轮廓
    for (int _i = 0; _i < num_contours; _i++) {
        float ssid;
        // 获取最小外接矩形
        cv::RotatedRect box = cv::minAreaRect(contours[_i]);
        auto array = get_mini_boxes(box, ssid);

        auto box_for_unclip = array;

        // 如果最小外接矩形的边长小于最小尺寸，则跳过
        if (ssid < min_size) {
            continue;
        }

        float score;
        // 计算边界框得分
        score = box_score_fast(array, pred);

        if (score < box_thresh) {
            continue;
        }

        // 对未裁剪的边界框进行处理
        cv::RotatedRect points = unclip(box_for_unclip);

        cv::RotatedRect clipbox = points;
        auto cliparray = get_mini_boxes(clipbox, ssid);

        if (ssid < min_size + 2)
            continue;

        int dest_width = pred.cols;
        int dest_height = pred.rows;
        std::vector<std::vector<int>> intcliparray;

        // 对裁剪后的边界框坐标进行处理
        for (int num_pt = 0; num_pt < 4; num_pt++) {
            std::vector<int> a{int(clampf(roundf(cliparray[num_pt][0] / float(width) * float(dest_width)), 0, float(dest_width)),
                               int(clampf(roundf(cliparray[num_pt][1] / float(height) * float(dest_height)), 0, float(dest_height))};
            intcliparray.emplace_back(std::move(a));
        }
        boxes.emplace_back(std::move(intcliparray));
    } // 结束轮廓遍历
    return boxes;
}

// 返回两个整数中的最大值
int _max(int a, int b) { return a >= b ? a : b; }
# 定义一个函数，返回两个整数中的最小值
int _min(int a, int b) { return a >= b ? b : a; }

# 对检测结果进行过滤，根据比例缩放框的坐标，并根据原始图像大小进行裁剪
std::vector<std::vector<std::vector<int>>>
filter_tag_det_res(const std::vector<std::vector<std::vector<int>>> &o_boxes,
                   float ratio_h, float ratio_w, const cv::Mat &srcimg) {
  # 获取原始图像的高度和宽度
  int oriimg_h = srcimg.rows;
  int oriimg_w = srcimg.cols;
  # 复制输入的框信息
  std::vector<std::vector<std::vector<int>>> boxes{o_boxes};
  # 存储处理后的框信息
  std::vector<std::vector<std::vector<int>>> root_points;
  
  # 遍历每个框
  for (int n = 0; n < boxes.size(); n++) {
    # 对框的顶点按顺时针排序
    boxes[n] = order_points_clockwise(boxes[n]);
    # 对每个顶点进行比例缩放
    for (int m = 0; m < boxes[0].size(); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      # 对坐标进行裁剪，确保在原始图像范围内
      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  # 遍历每个框，计算框的宽度和高度，将符合条件的框加入结果中
  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                          pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                           pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 10 || rect_height <= 10)
      continue;
    root_points.push_back(boxes[n]);
  }
  # 返回处理后的框信息
  return root_points;
}
```