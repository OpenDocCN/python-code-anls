# `.\yolov8\examples\YOLOv8-CPP-Inference\inference.cpp`

```py
// 引入推断头文件 "inference.h"
#include "inference.h"

// 构造函数，接收 ONNX 模型路径、模型输入形状、类别文件路径和是否使用 CUDA
Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    // 将参数赋值给成员变量
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    // 载入 ONNX 网络模型
    loadOnnxNetwork();
    // loadClassesFromFile(); 此处的类别已经在代码中硬编码，不需要再从文件加载
}

// 运行推断的函数，接收输入图像并返回检测结果向量
std::vector<Detection> Inference::runInference(const cv::Mat &input)
{
    // 将输入图像复制给模型输入变量
    cv::Mat modelInput = input;
    // 如果设置了 letterBoxForSquare 并且模型形状是正方形，则将输入图像格式化为正方形
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    // 将输入图像转换为 blob 格式
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    // 设置网络输入
    net.setInput(blob);

    // 前向传播得到网络输出
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // 获取输出张量的维度信息
    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    // 初始化是否为 YOLOv8 模型的标志
    bool yolov8 = false;
    // 如果维度信息表明是 YOLOv8 模型
    if (dimensions > rows) // 检查 shape[2] 是否大于 shape[1]（YOLOv8）
    {
        yolov8 = true;
        // 更新 rows 和 dimensions 的值
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        // 调整输出张量的形状以适应 YOLOv8 的要求
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    // 获取输出数据的指针
    float *data = (float *)outputs[0].data;

    // 计算图像缩放因子
    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    // 初始化类别、置信度和边界框向量
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 遍历每个输出行
    for (int i = 0; i < rows; ++i)
    {
        // 如果是 YOLOv8 模型
        if (yolov8)
        {
            // 提取类别分数数据的指针
            float *classes_scores = data + 4;
    
            // 创建一个 OpenCV Mat 对象来存储类别分数
            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            
            // 存储最大类别分数及其对应的类别 ID
            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
    
            // 如果最大类别分数高于模型分数阈值
            if (maxClassScore > modelScoreThreshold)
            {
                // 将最大类别分数和类别 ID 添加到结果中
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
    
                // 提取检测框的位置和尺寸信息
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
    
                // 计算检测框的左上角坐标及宽高
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
    
                // 将检测框信息添加到 boxes 向量中
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // 如果是 YOLOv5 模型
        {
            // 提取置信度数据
            float confidence = data[4];
    
            // 如果置信度高于模型置信度阈值
            if (confidence >= modelConfidenceThreshold)
            {
                // 提取类别分数数据的指针
                float *classes_scores = data + 5;
    
                // 创建一个 OpenCV Mat 对象来存储类别分数
                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
    
                // 存储最大类别分数及其对应的类别 ID
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
    
                // 如果最大类别分数高于模型分数阈值
                if (max_class_score > modelScoreThreshold)
                {
                    // 将最大置信度和类别 ID 添加到结果中
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
    
                    // 提取检测框的位置和尺寸信息
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
    
                    // 计算检测框的左上角坐标及宽高
                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
    
                    // 将检测框信息添加到 boxes 向量中
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    
        // 更新数据指针到下一个检测结果
        data += dimensions;
    }
    
    // 对检测到的所有边界框执行非最大抑制操作，得到最终的边界框索引
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
    
    // 创建一个空的 Detection 结构体向量，用于存储最终的检测结果
    std::vector<Detection> detections{};
    
    // 遍历经过非最大抑制后的边界框索引
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        // 获取当前边界框的索引
        int idx = nms_result[i];
    
        // 创建一个 Detection 对象来存储当前边界框的检测结果
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
    
        // 生成一个随机颜色作为当前类别的显示颜色
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));
    
        // 获取当前类别的名称
        result.className = classes[result.class_id];
    
        // 将边界框信息添加到 Detection 对象中
        result.box = boxes[idx];
    
        // 将当前检测结果添加到 detections 向量中
        detections.push_back(result);
    }
    
    // 返回最终的检测结果向量
    return detections;
}

void Inference::loadClassesFromFile()
{
    // 打开指定路径的文件流
    std::ifstream inputFile(classesPath);
    // 如果文件成功打开
    if (inputFile.is_open())
    {
        // 用于存储每行读取的类别名称的变量
        std::string classLine;
        // 循环读取每行数据，将其添加到类别列表中
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        // 关闭文件流
        inputFile.close();
    }
}

void Inference::loadOnnxNetwork()
{
    // 从 ONNX 模型路径读取神经网络
    net = cv::dnn::readNetFromONNX(modelPath);
    // 如果启用了 CUDA 加速
    if (cudaEnabled)
    {
        // 输出提示信息，表明正在使用 CUDA 运行
        std::cout << "\nRunning on CUDA" << std::endl;
        // 设置神经网络的后端为 CUDA
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // 设置目标设备为 CUDA
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        // 输出提示信息，表明正在使用 CPU 运行
        std::cout << "\nRunning on CPU" << std::endl;
        // 设置神经网络的后端为 OpenCV
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // 设置目标设备为 CPU
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    // 获取输入图像的列数和行数
    int col = source.cols;
    int row = source.rows;
    // 计算出输入图像中较大的维度
    int _max = MAX(col, row);
    // 创建一个指定大小的全零图像作为结果
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    // 将输入图像复制到结果图像的左上角区域
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    // 返回格式化后的正方形图像
    return result;
}
```