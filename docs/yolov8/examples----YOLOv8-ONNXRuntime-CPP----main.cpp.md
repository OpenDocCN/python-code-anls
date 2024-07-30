# `.\yolov8\examples\YOLOv8-ONNXRuntime-CPP\main.cpp`

```py
void Detector(YOLO_V8*& p) {
    // 获取当前工作目录的路径
    std::filesystem::path current_path = std::filesystem::current_path();
    // 设置图像目录路径为当前路径下的 "images" 子目录
    std::filesystem::path imgs_path = current_path / "images";
    
    // 遍历图像目录中的每一个文件
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        // 检查文件扩展名是否为 .jpg, .png 或 .jpeg
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            // 获取图像文件的完整路径
            std::string img_path = i.path().string();
            // 读取图像文件为 OpenCV 的 Mat 对象
            cv::Mat img = cv::imread(img_path);
            
            // 定义存储推断结果的向量
            std::vector<DL_RESULT> res;
            // 使用 YOLO_V8 模型进行推断
            p->RunSession(img, res);

            // 遍历每一个推断结果
            for (auto& re : res)
            {
                // 生成随机颜色
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                // 在图像上绘制带有边界框的矩形
                cv::rectangle(img, re.box, color, 3);

                // 格式化置信度并生成标签
                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                // 在矩形上方绘制带有标签的矩形
                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                // 在带有标签的矩形上绘制标签文字
                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );
            }

            // 显示处理结果并等待按键
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}

void Classifier(YOLO_V8*& p)
{
    // 获取当前工作目录的路径
    std::filesystem::path current_path = std::filesystem::current_path();
    // 将图像目录路径设置为当前路径
    std::filesystem::path imgs_path = current_path;// / "images"
    
    // 生成随机设备并初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    
    // 遍历图像目录中的每一个文件
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        // 检查文件扩展名是否为 ".jpg" 或 ".png"
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
        {
            // 获取图像文件的路径
            std::string img_path = i.path().string();
            // 使用 OpenCV 读取图像文件
            cv::Mat img = cv::imread(img_path);
            // 创建用于存储深度学习模型结果的向量
            std::vector<DL_RESULT> res;
            // 调用某个模型的 RunSession 方法，分析图像并返回结果
            char* ret = p->RunSession(img, res);
    
            // 初始化文本输出位置的 Y 坐标
            float positionY = 50;
            // 遍历每个模型返回的结果
            for (int i = 0; i < res.size(); i++)
            {
                // 随机生成颜色值
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                // 在图像上添加分类标签
                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                // 在图像上添加分类结果的置信度
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                // 更新下一个文本输出的 Y 坐标位置
                positionY += 50;
            }
    
            // 显示带有分类信息的图像窗口
            cv::imshow("TEST_CLS", img);
            // 等待用户按下键盘上的任意键
            cv::waitKey(0);
            // 关闭所有打开的图像窗口
            cv::destroyAllWindows();
            // 将处理后的图像保存到指定路径（注释掉的代码）
            //cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
        }
    }
}

// 读取 coco.yaml 文件，提取类别名称并存储到 p->classes 中
int ReadCocoYaml(YOLO_V8*& p) {
    // 打开 YAML 文件
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // 逐行读取文件内容
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // 查找 names 部分的起始和结束位置
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // 提取类别名称
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // 提取分隔符前的内容（数字）
        std::getline(ss, name); // 提取分隔符后的内容（字符串）
        names.push_back(name);
    }

    // 将类别名称存储到 p->classes 中
    p->classes = names;
    return 0;
}


// 进行检测测试
void DetectTest()
{
    // 创建 YOLO_V8 对象指针
    YOLO_V8* yoloDetector = new YOLO_V8;
    // 读取 coco.yaml 文件中的类别名称
    ReadCocoYaml(yoloDetector);
    // 设置检测参数
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "yolov8n.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // 使用 GPU 进行 FP32 推断
    params.modelType = YOLO_DETECT_V8;
    // 使用 GPU 进行 FP16 推断（注意：需要更改为 fp16 的 onnx 模型）
    // params.modelType = YOLO_DETECT_V8_HALF;

#else
    // 使用 CPU 进行推断
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    // 创建推断会话
    yoloDetector->CreateSession(params);
    // 进行检测
    Detector(yoloDetector);
}


// 进行分类测试
void ClsTest()
{
    // 创建 YOLO_V8 对象指针
    YOLO_V8* yoloDetector = new YOLO_V8;
    // 设置模型路径
    std::string model_path = "cls.onnx";
    // 读取 coco.yaml 文件中的类别名称
    ReadCocoYaml(yoloDetector);
    // 设置分类器参数
    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
    // 创建分类器会话
    yoloDetector->CreateSession(params);
    // 进行分类
    Classifier(yoloDetector);
}


// 主函数
int main()
{
    // 执行分类测试
    ClsTest();
}
```