# `.\PaddleOCR\deploy\cpp_infer\src\postprocess_op.cpp`

```py
// 包含 PaddleOCR 的后处理操作的头文件
#include <include/postprocess_op.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// 计算轮廓的面积
void DBPostProcessor::GetContourArea(const std::vector<std::vector<float>> &box,
                                     float unclip_ratio, float &distance) {
  // 定义点的数量
  int pts_num = 4;
  // 初始化面积和距离
  float area = 0.0f;
  float dist = 0.0f;
  // 遍历每个点
  for (int i = 0; i < pts_num; i++) {
    // 计算面积
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    // 计算距离
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  // 计算绝对值的面积
  area = fabs(float(area / 2.0));

  // 计算距离
  distance = area * unclip_ratio / dist;
}

// 对文本框进行解除裁剪
cv::RotatedRect DBPostProcessor::UnClip(std::vector<std::vector<float>> box,
                                        const float &unclip_ratio) {
  // 初始化距离
  float distance = 1.0;

  // 计算距离
  GetContourArea(box, unclip_ratio, distance);

  // 创建 ClipperOffset 对象
  ClipperLib::ClipperOffset offset;
  // 创建路径对象
  ClipperLib::Path p;
  // 添加路径点
  p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
    << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
    << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
    // 创建一个 ClipperLib::IntPoint 对象，使用 box[3][0] 和 box[3][1] 的整数部分作为坐标
    ClipperLib::IntPoint p(int(box[3][0]), int(box[3][1]));
    // 将路径添加到 offset 中，采用 jtRound 模式，表示路径为封闭多边形
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    // 创建一个 ClipperLib::Paths 对象 soln
    ClipperLib::Paths soln;
    // 执行 offset 操作，将结果存储在 soln 中，distance 为偏移距离
    offset.Execute(soln, distance);
    // 创建一个存储 cv::Point2f 的向量 points

    std::vector<cv::Point2f> points;

    // 遍历 soln 中的路径
    for (int j = 0; j < soln.size(); j++) {
        // 遍历当前路径中的点
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            // 将当前点的 X 和 Y 坐标作为 cv::Point2f 对象的坐标，添加到 points 中
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    // 创建一个 cv::RotatedRect 对象 res

    cv::RotatedRect res;
    // 如果 points 中没有点，则创建一个默认的旋转矩形
    if (points.size() <= 0) {
        res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
    } else {
        // 否则，根据 points 中的点，计算最小外接矩形
        res = cv::minAreaRect(points);
    }
    // 返回最终的旋转矩形结果
    return res;
// 将 OpenCV 的 Mat 类型转换为二维浮点型数组
float **DBPostProcessor::Mat2Vec(cv::Mat mat) {
  // 创建一个二维浮点型数组
  auto **array = new float *[mat.rows];
  // 遍历每一行，为每一行创建一个一维浮点型数组
  for (int i = 0; i < mat.rows; ++i)
    array[i] = new float[mat.cols];
  // 将 Mat 类型中的数据复制到二维数组中
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

// 对点进行顺时针排序
std::vector<std::vector<int>>
DBPostProcessor::OrderPointsClockwise(std::vector<std::vector<int>> pts) {
  // 复制传入的点集
  std::vector<std::vector<int>> box = pts;
  // 按照 x 坐标排序
  std::sort(box.begin(), box.end(), XsortInt);

  // 找到最左边的两个点和最右边的两个点
  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  // 确保左边的点按照 y 坐标升序排列
  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  // 确保右边的点按照 y 坐标升序排列
  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  // 组成矩形的四个点，按照左上、右上、右下、左下的顺序返回
  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

// 将 OpenCV 的 Mat 类型转换为二维浮点型向量
std::vector<std::vector<float>> DBPostProcessor::Mat2Vector(cv::Mat mat) {
  // 创建一个二维浮点型向量
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  // 遍历每一行，将每一行数据存储为一维浮点型向量，再将其添加到二维向量中
  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

// 按照浮点数的第一个元素进行升序排序
bool DBPostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

// 按照整数的第一个元素进行升序排序
bool DBPostProcessor::XsortInt(std::vector<int> a, std::vector<int> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

// 返回二维浮点型向量
std::vector<std::vector<float>>
// 计算旋转矩形的最大边长作为 ssid，并赋值给传入的引用参数
DBPostProcessor::GetMiniBoxes(cv::RotatedRect box, float &ssid) {
  ssid = std::max(box.size.width, box.size.height);

  // 获取旋转矩形的四个顶点坐标
  cv::Mat points;
  cv::boxPoints(box, points);

  // 将顶点坐标转换为数组，并按 x 坐标排序
  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  // 根据顶点坐标的 y 坐标大小关系重新排序
  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
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

  // 更新顶点坐标数组的顺序
  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  // 返回更新后的顶点坐标数组
  return array;
}

// 计算多边形得分
float DBPostProcessor::PolygonScoreAcc(std::vector<cv::Point> contour,
                                       cv::Mat pred) {
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (int i = 0; i < contour.size(); ++i) {
    box_x.push_back(contour[i].x);
    box_y.push_back(contour[i].y);
  }

  // 计算多边形的边界范围
  int xmin =
      clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int xmax =
      clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int ymin =
      clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
            height - 1);
  int ymax =
      clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
            height - 1);

  // 创建一个与多边形边界范围相匹配的掩码
  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point *rook_point = new cv::Point[contour.size()];

  for (int i = 0; i < contour.size(); ++i) {
    // 将每个角点的坐标减去最小坐标，得到相对于最小坐标的偏移量，存储在 rook_point 中
    rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
  }
  // 定义一个指向 rook_point 的指针数组
  const cv::Point *ppt[1] = {rook_point};
  // 定义一个整型数组，存储轮廓的大小
  int npt[] = {int(contour.size())};

  // 使用指定的多边形填充 mask，填充颜色为 Scalar(1)
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  // 创建一个 croppedImg，从 pred 中裁剪出指定区域的图像
  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  // 计算裁剪后图像在 mask 区域内的平均值，存储在 score 中
  float score = cv::mean(croppedImg, mask)[0];

  // 释放 rook_point 数组的内存
  delete[] rook_point;
  // 返回计算得到的 score
  return score;
// 计算包围框得分的快速方法
float DBPostProcessor::BoxScoreFast(std::vector<std::vector<float>> box_array,
                                    cv::Mat pred) {
  // 复制输入的包围框数组
  auto array = box_array;
  // 获取预测矩阵的宽度和高度
  int width = pred.cols;
  int height = pred.rows;

  // 提取包围框的 x 和 y 坐标
  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  // 计算包围框的 xmin、xmax、ymin、ymax
  int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4))), 0,
                   width - 1);
  int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4))), 0,
                   height - 1);

  // 创建一个 mask 矩阵
  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  // 计算包围框的四个顶点坐标
  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  // 在 mask 上填充多边形
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  // 从预测矩阵中裁剪出感兴趣区域
  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  // 计算裁剪后区域的平均值作为得分
  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

// 从位图中获取包围框
std::vector<std::vector<std::vector<int>>> DBPostProcessor::BoxesFromBitmap(
    const cv::Mat pred, const cv::Mat bitmap, const float &box_thresh,
  // 定义最小尺寸和最大候选框数量
  const float &det_db_unclip_ratio, const std::string &det_db_score_mode) {
  const int min_size = 3;
  const int max_candidates = 1000;

  // 获取位图的宽度和高度
  int width = bitmap.cols;
  int height = bitmap.rows;

  // 定义轮廓和层次结构
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  // 查找位图的轮廓
  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  // 确定要处理的轮廓数量
  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  // 定义包含候选框的三维向量
  std::vector<std::vector<std::vector<int>>> boxes;

  // 遍历每个轮廓
  for (int _i = 0; _i < num_contours; _i++) {
    // 如果轮廓点数小于等于2，则跳过
    if (contours[_i].size() <= 2) {
      continue;
    }
    // 定义旋转矩形和ssid
    float ssid;
    cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    auto array = GetMiniBoxes(box, ssid);

    auto box_for_unclip = array;

    // 如果ssid小于最小尺寸，则跳过
    if (ssid < min_size) {
      continue;
    }

    // 计算得分
    float score;
    if (det_db_score_mode == "slow")
      /* compute using polygon*/
      score = PolygonScoreAcc(contours[_i], pred);
    else
      score = BoxScoreFast(array, pred);

    // 如果得分低于阈值，则跳过
    if (score < box_thresh)
      continue;

    // 对候选框进行解除裁剪
    cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
    // 如果解除裁剪后的高度和宽度小于1.001，则跳过
    if (points.size.height < 1.001 && points.size.width < 1.001) {
      continue;
    }

    // 定义裁剪框和裁剪后的候选框
    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);

    // 如果ssid小于最小尺寸加2，则跳过
    if (ssid < min_size + 2)
      continue;

    // 获取预测图像的宽度和高度
    int dest_width = pred.cols;
    int dest_height = pred.rows;
    // 定义整数类型的裁剪后的候选框
    std::vector<std::vector<int>> intcliparray;
    // 循环4次，分别处理4个点
    for (int num_pt = 0; num_pt < 4; num_pt++) {
      // 计算每个点的新坐标，并将其存储在向量a中
      std::vector<int> a{int(clampf(roundf(cliparray[num_pt][0] / float(width) *
                                           float(dest_width)),
                                    0, float(dest_width))),
                         int(clampf(roundf(cliparray[num_pt][1] /
                                           float(height) * float(dest_height)),
                                    0, float(dest_height)))};
      // 将向量a添加到intcliparray中
      intcliparray.push_back(a);
    }
    // 将intcliparray添加到boxes中
    boxes.push_back(intcliparray);

  } // 结束for循环
  // 返回boxes向量
  return boxes;
// 过滤标签检测结果，根据比例和原始图像大小对边界框进行处理
std::vector<std::vector<std::vector<int>>> DBPostProcessor::FilterTagDetRes(
    std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
    float ratio_w, cv::Mat srcimg) {
  // 获取原始图像的高度和宽度
  int oriimg_h = srcimg.rows;
  int oriimg_w = srcimg.cols;

  // 存储处理后的边界框的根点
  std::vector<std::vector<std::vector<int>>> root_points;
  // 遍历边界框
  for (int n = 0; n < boxes.size(); n++) {
    // 对边界框的顶点按顺时针排序
    boxes[n] = OrderPointsClockwise(boxes[n]);
    // 对每个顶点进行比例缩放
    for (int m = 0; m < boxes[0].size(); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      // 限制顶点坐标在图像范围内
      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  // 根据边界框的宽高计算矩形的宽高，过滤掉宽高小于等于4的矩形
  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width = int(sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                          pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = int(sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                           pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 4 || rect_height <= 4)
      continue;
    // 将符合条件的矩形添加到根点列表中
    root_points.push_back(boxes[n]);
  }
  return root_points;
}

// 初始化表格后处理器，读取标签文件并根据参数进行初始化
void TablePostProcessor::init(std::string label_path,
                              bool merge_no_span_structure) {
  // 读取标签文件内容到标签列表
  this->label_list_ = Utility::ReadDict(label_path);
  // 如果需要合并无跨度结构，则添加特殊标签并处理标签列表
  if (merge_no_span_structure) {
    this->label_list_.push_back("<td></td>");
    // 遍历标签列表，删除特定标签
    std::vector<std::string>::iterator it;
    for (it = this->label_list_.begin(); it != this->label_list_.end();) {
      if (*it == "<td>") {
        it = this->label_list_.erase(it);
      } else {
        ++it;
      }
    }
  }
  // 添加特殊字符到标签列表的开头和结尾
  this->label_list_.insert(this->label_list_.begin(), this->beg);
  this->label_list_.push_back(this->end);
}

// 运行表格后处理器，处理定位预测、结构概率、识别分数等数据
void TablePostProcessor::Run(
    std::vector<float> &loc_preds, std::vector<float> &structure_probs,
    std::vector<float> &rec_scores, std::vector<int> &loc_preds_shape,
    std::vector<int> &structure_probs_shape,
    // 传入的参数包括批量的 HTML 标签、框坐标、宽度列表和高度列表
    std::vector<std::vector<std::string>> &rec_html_tag_batch,
    std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
    std::vector<int> &width_list, std::vector<int> &height_list) {
  // 遍历结构概率形状的第一个维度，即批量索引
  for (int batch_idx = 0; batch_idx < structure_probs_shape[0]; batch_idx++) {
    // image tags and boxs
    // 创建存储 HTML 标签和框坐标的容器
    std::vector<std::string> rec_html_tags;
    std::vector<std::vector<int>> rec_boxes;

    // 初始化得分、计数、字符得分和字符索引
    float score = 0.f;
    int count = 0;
    float char_score = 0.f;
    int char_idx = 0;

    // 步骤
    // 遍历每个步骤的索引
    for (int step_idx = 0; step_idx < structure_probs_shape[1]; step_idx++) {
      // 定义 HTML 标签和记录框
      std::string html_tag;
      std::vector<int> rec_box;
      
      // 获取当前步骤的起始索引
      int step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) * structure_probs_shape[2];
      
      // 获取当前步骤的字符索引和字符分数
      char_idx = int(Utility::argmax(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      char_score = float(*std::max_element(
          &structure_probs[step_start_idx],
          &structure_probs[step_start_idx + structure_probs_shape[2]]));
      html_tag = this->label_list_[char_idx];

      // 如果当前步骤大于0且 HTML 标签为结束标签，则跳出循环
      if (step_idx > 0 && html_tag == this->end) {
        break;
      }
      // 如果 HTML 标签为起始标签，则继续下一个步骤
      if (html_tag == this->beg) {
        continue;
      }
      
      // 计数和分数累加
      count += 1;
      score += char_score;
      rec_html_tags.push_back(html_tag);

      // 处理框
      if (html_tag == "<td>" || html_tag == "<td" || html_tag == "<td></td>") {
        // 遍历每个点的索引
        for (int point_idx = 0; point_idx < loc_preds_shape[2]; point_idx++) {
          step_start_idx = (batch_idx * structure_probs_shape[1] + step_idx) * loc_preds_shape[2] + point_idx;
          float point = loc_preds[step_start_idx];
          // 根据点的索引计算坐标值
          if (point_idx % 2 == 0) {
            point = int(point * width_list[batch_idx]);
          } else {
            point = int(point * height_list[batch_idx]);
          }
          rec_box.push_back(point);
        }
        rec_boxes.push_back(rec_box);
      }
    }
    
    // 计算平均分数
    score /= count;
    // 如果分数为 NaN 或者记录框为空，则将分数设为-1
    if (std::isnan(score) || rec_boxes.size() == 0) {
      score = -1;
    }
    // 将分数、记录框和 HTML 标签添加到对应的批次中
    rec_scores.push_back(score);
    rec_boxes_batch.push_back(rec_boxes);
    rec_html_tag_batch.push_back(rec_html_tags);
  }
// 初始化后处理器，设置标签路径、得分阈值、NMS阈值和FPN步长
void PicodetPostProcessor::init(std::string label_path,
                                const double score_threshold,
                                const double nms_threshold,
                                const std::vector<int> &fpn_stride) {
  // 读取标签字典
  this->label_list_ = Utility::ReadDict(label_path);
  // 设置得分阈值
  this->score_threshold_ = score_threshold;
  // 设置NMS阈值
  this->nms_threshold_ = nms_threshold;
  // 获取类别数量
  this->num_class_ = label_list_.size();
  // 设置FPN步长
  this->fpn_stride_ = fpn_stride;
}

// 运行后处理器，处理预测结果
void PicodetPostProcessor::Run(std::vector<StructurePredictResult> &results,
                               std::vector<std::vector<float>> outs,
                               std::vector<int> ori_shape,
                               std::vector<int> resize_shape, int reg_max) {
  // 获取输入图像的高度和宽度
  int in_h = resize_shape[0];
  int in_w = resize_shape[1];
  // 计算高度和宽度的缩放因子
  float scale_factor_h = resize_shape[0] / float(ori_shape[0]);
  float scale_factor_w = resize_shape[1] / float(ori_shape[1]);

  // 初始化存储每个类别的边界框结果
  std::vector<std::vector<StructurePredictResult>> bbox_results;
  bbox_results.resize(this->num_class_);
  // 遍历FPN步长
  for (int i = 0; i < this->fpn_stride_.size(); ++i) {
    // 计算特征图的高度和宽度
    int feature_h = std::ceil((float)in_h / this->fpn_stride_[i]);
    int feature_w = std::ceil((float)in_w / this->fpn_stride_[i]);
    // 遍历特征图中的每个像素点
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      // 初始化得分和标签
      float score = 0;
      int cur_label = 0;
      // 遍历每个类别
      for (int label = 0; label < this->num_class_; label++) {
        // 更新最高得分和对应的标签
        if (outs[i][idx * this->num_class_ + label] > score) {
          score = outs[i][idx * this->num_class_ + label];
          cur_label = label;
        }
      }
      // 如果得分高于阈值
      if (score > this->score_threshold_) {
        // 计算行列索引
        int row = idx / feature_w;
        int col = idx % feature_w;
        // 提取边界框预测值
        std::vector<float> bbox_pred(
            outs[i + this->fpn_stride_.size()].begin() + idx * 4 * reg_max,
            outs[i + this->fpn_stride_.size()].begin() +
                (idx + 1) * 4 * reg_max);
        // 将边界框结果添加到对应类别的结果中
        bbox_results[cur_label].push_back(
            this->disPred2Bbox(bbox_pred, cur_label, score, col, row,
                               this->fpn_stride_[i], resize_shape, reg_max));
      }
    }
  }
  // 遍历每个类别的边界框结果
  for (int i = 0; i < bbox_results.size(); i++) {
    // 检查是否该类别没有边界框结果
    bool flag = bbox_results[i].size() <= 0;
  }
  // 再次遍历每个类别的边界框结果
  for (int i = 0; i < bbox_results.size(); i++) {
    // 检查是否该类别没有边界框结果
    bool flag = bbox_results[i].size() <= 0;
    // 如果该类别没有边界框结果，则跳过
    if (bbox_results[i].size() <= 0) {
      continue;
    }
    // 对该类别的边界框结果进行非极大值抑制
    this->nms(bbox_results[i], this->nms_threshold_);
    // 对每个边界框进行缩放
    for (auto box : bbox_results[i]) {
      box.box[0] = box.box[0] / scale_factor_w;
      box.box[2] = box.box[2] / scale_factor_w;
      box.box[1] = box.box[1] / scale_factor_h;
      box.box[3] = box.box[3] / scale_factor_h;
      // 将缩放后的边界框结果添加到最终结果中
      results.push_back(box);
    }
  }
}

// 将预测的距离转换为边界框坐标
StructurePredictResult PicodetPostProcessor::disPred2Bbox(std::vector<float> bbox_pred, int label,
                                   float score, int x, int y, int stride,
                                   std::vector<int> im_shape, int reg_max) {
  // 计算中心点坐标
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  // 初始化距离预测结果
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  // 遍历四个方向
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    // 获取当前方向的距离预测值
    std::vector<float> bbox_pred_i(bbox_pred.begin() + i * reg_max,
                                   bbox_pred.begin() + (i + 1) * reg_max);
    // 对距离预测值进行 softmax 激活函数处理
    std::vector<float> dis_after_sm =
        Utility::activation_function_softmax(bbox_pred_i);
    // 计算距离
    for (int j = 0; j < reg_max; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
  }

  // 计算边界框的坐标
  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)im_shape[1]);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)im_shape[0]);

  // 创建结构预测结果对象
  StructurePredictResult result_item;
  result_item.box = {xmin, ymin, xmax, ymax};
  result_item.type = this->label_list_[label];
  result_item.confidence = score;

  return result_item;
}

// 非极大值抑制
void PicodetPostProcessor::nms(std::vector<StructurePredictResult> &input_boxes,
                               float nms_threshold) {
  // 根据置信度对边界框进行排序
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](StructurePredictResult a, StructurePredictResult b) {
              return a.confidence > b.confidence;
            });
  // 初始化标记数组
  std::vector<int> picked(input_boxes.size(), 1);

  // 遍历边界框
  for (int i = 0; i < input_boxes.size(); ++i) {
    if (picked[i] == 0) {
      continue;
    }
    for (int j = i + 1; j < input_boxes.size(); ++j) {
      if (picked[j] == 0) {
        continue;
      }
      // 计算 IoU
      float iou = Utility::iou(input_boxes[i].box, input_boxes[j].box);
      // 根据阈值进行非极大值抑制
      if (iou > nms_threshold) {
        picked[j] = 0;
      }
    }
  }
  // 创建一个新的向量用于存储经过非极大值抑制处理后的输入框
  std::vector<StructurePredictResult> input_boxes_nms;
  // 遍历原始输入框
  for (int i = 0; i < input_boxes.size(); ++i) {
    // 如果该输入框被选中
    if (picked[i] == 1) {
      // 将该输入框添加到经过非极大值抑制处理后的输入框向量中
      input_boxes_nms.push_back(input_boxes[i]);
    }
  }
  // 将原始输入框替换为经过非极大值抑制处理后的输入框向量
  input_boxes = input_boxes_nms;
}
``` 


} // namespace PaddleOCR
```