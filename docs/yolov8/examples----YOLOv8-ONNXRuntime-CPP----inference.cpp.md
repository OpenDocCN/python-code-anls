# `.\yolov8\examples\YOLOv8-ONNXRuntime-CPP\inference.cpp`

```py
#include "inference.h"
#include <regex>

// 定义预处理时的性能基准
#define benchmark
// 定义比较两个数中较小的宏
#define min(a,b)            (((a) < (b)) ? (a) : (b))

// YOLO_V8 类的默认构造函数
YOLO_V8::YOLO_V8() {

}

// YOLO_V8 类的析构函数，释放 session 资源
YOLO_V8::~YOLO_V8() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    // 将 half 类型映射为 ONNX 中的 FLOAT16 数据类型
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

// 从图像生成 Blob 数据，用于模型输入
template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    // 遍历图像像素并生成 Blob 数据
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                // 归一化像素值到 [0, 1] 并存入 Blob 中
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;  // 返回处理成功标志
}

// YOLO_V8 类的图像预处理函数
char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        // 如果图像是三通道的，进行颜色空间转换为 RGB
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        // 如果图像不是三通道的，先转换为三通道的灰度图，再转换为 RGB
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    // 根据模型类型选择不同的预处理方式
    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_POSE:
    case YOLO_DETECT_V8_HALF:
    case YOLO_POSE_V8_HALF: // LetterBox 模式
    {
        if (iImg.cols >= iImg.rows)
        {
            // 根据宽度缩放比例调整图像大小，并保持宽度不变，高度按比例缩放
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            // 根据高度缩放比例调整图像大小，并保持高度不变，宽度按比例缩放
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        // 创建一个与目标图像大小相同的临时图像，并将处理后的图像复制到其中
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg; // 将处理后的图像赋给输出图像
        break;
    }
    case YOLO_CLS: // CenterCrop 模式
    {
        int h = iImg.rows;
        int w = iImg.cols;
        int m = min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        // 中心裁剪图像，并调整大小到指定尺寸
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
    return RET_OK; // 返回处理成功标志
}

// YOLO_V8 类的创建会话函数，用于初始化模型会话
char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    // 检查模型路径中是否包含中文字符
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        // 如果模型路径包含中文字符，返回错误信息
        Ret = "[YOLO_V8]:Your model path is error.Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        // 从输入参数 iParams 中获取 rectConfidenceThreshold 的值
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        // 从输入参数 iParams 中获取 iouThreshold 的值
        iouThreshold = iParams.iouThreshold;
        // 从输入参数 iParams 中获取 imgSize 的值
        imgSize = iParams.imgSize;
        // 从输入参数 iParams 中获取 modelType 的值
        modelType = iParams.modelType;
        // 创建 Ort 环境对象，设置日志级别为警告级，名称为 "Yolo"
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        // 创建 Ort 会话选项对象
        Ort::SessionOptions sessionOption;
        // 如果 iParams 中启用了 CUDA 加速
        if (iParams.cudaEnable)
        {
            // 将 cudaEnable 标志设为 true
            cudaEnable = iParams.cudaEnable;
            // 创建 OrtCUDAProviderOptions 对象，指定设备 ID 为 0
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            // 将 CUDA 加速选项附加到会话选项中
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        // 设置图优化级别为启用全部优化
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // 设置会话内操作的线程数目
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        // 设置日志严重级别
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);
#ifdef _WIN32
        // 计算将 UTF-8 编码转换为宽字符所需的缓冲区大小
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        // 分配用于存储宽字符路径的内存
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        // 将 UTF-8 编码转换为宽字符
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        // 确保宽字符路径以 null 结尾
        wide_cstr[ModelPathSize] = L'\0';
        // 将宽字符路径赋值给 modelPath
        const wchar_t* modelPath = wide_cstr;
#else
        // 在非 Windows 平台，直接使用 UTF-8 编码的 modelPath
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        // 创建 Ort::Session 对象，加载模型
        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        // 获取模型输入节点的数量
        size_t inputNodesNum = session->GetInputCount();
        // 遍历输入节点，获取每个输入节点的名称，并存储为 C 字符串
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            std::strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        // 获取模型输出节点的数量
        size_t OutputNodesNum = session->GetOutputCount();
        // 遍历输出节点，获取每个输出节点的名称，并存储为 C 字符串
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            std::strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        // 初始化 Ort::RunOptions 对象为 nullptr
        options = Ort::RunOptions{ nullptr };
        // 预热模型会话
        WarmUpSession();
        // 返回成功标志
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        // 捕获异常并组合错误消息
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        // 分配内存并复制组合后的错误消息
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        // 打印错误消息
        std::cout << merged << std::endl;
        // 释放内存
        delete[] merged;
        // 返回错误消息
        return "[YOLO_V8]:Create session failed.";
    }

}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark

    // 初始化返回值为成功标志
    char* Ret = RET_OK;
    cv::Mat processedImg;
    // 对输入图像进行预处理
    PreProcess(iImg, imgSize, processedImg);
    // 根据模型类型执行不同的处理逻辑
    if (modelType < 4)
    {
        // 分配 float 类型的 blob 存储图像数据
        float* blob = new float[processedImg.total() * 3];
        // 将图像转换为 blob
        BlobFromImage(processedImg, blob);
        // 定义输入节点的维度
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        // 处理张量数据
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else
    {
#ifdef USE_CUDA
        // 在使用 CUDA 的情况下，分配 half 类型的 blob 存储图像数据
        half* blob = new half[processedImg.total() * 3];
        // 将图像转换为 blob
        BlobFromImage(processedImg, blob);
        // 定义输入节点的维度
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        // 处理张量数据
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    // 返回执行结果
    return Ret;
}


template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    # 使用 Ort::Value 类创建一个输入张量 inputTensor，其类型基于模板参数 N 的移除指针后的类型
    # Ort::Value::CreateTensor 是创建张量的方法
    # Ort::MemoryInfo::CreateCpu 用于指定内存信息，创建一个在 CPU 上分配内存的内存信息对象
    # OrtDeviceAllocator 是分配器对象，用于在指定设备上分配内存
    # OrtMemTypeCPU 表示内存类型为 CPU 内存
    # blob 是包含数据的指针或数组，用于初始化张量
    # 3 * imgSize.at(0) * imgSize.at(1) 是张量的总元素数量
    # inputNodeDims.data() 提供了张量的维度数据的指针，描述张量的形状
    # inputNodeDims.size() 是张量的维度数量
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    // 记录第二次时间戳，用于性能基准
    clock_t starttime_2 = clock();
#endif // benchmark
    // 运行推断会话，获取输出张量
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    // 记录第三次时间戳，用于性能基准
    clock_t starttime_3 = clock();
#endif // benchmark

    // 获取输出张量的类型信息
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    // 获取输出节点的维度信息
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    // 获取输出张量的可变数据指针
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    // 删除之前的 blob 内存
    delete[] blob;
    // 根据模型类型执行不同的处理
    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        // 获取第二维和第三维的大小，用于后续处理
        int strideNum = outputNodeDims[1];//8400
        int signalResultNum = outputNodeDims[2];//84
        // 初始化用于存储检测结果的向量和矩阵
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (modelType == YOLO_DETECT_V8)
        {
            // 如果模型类型是 YOLO_DETECT_V8，使用 FP32 类型创建原始数据矩阵
            rawData = cv::Mat(strideNum, signalResultNum, CV_32F, output);
        }
        else
        {
            // 如果模型类型是 YOLO_DETECT_V8_HALF，使用 FP16 类型创建原始数据矩阵，并转换为 FP32 类型
            rawData = cv::Mat(strideNum, signalResultNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        //Note:
        //ultralytics 添加转置操作以调整 yolov8 模型的输出，使其与 yolov8/v5/v7 的形状相同
        //https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
        //rowData = rowData.t();

        // 获取原始数据矩阵的数据指针
        float* data = (float*)rawData.data;

        // 遍历原始数据矩阵的每一行
        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            // 从原始数据中提取分数，并转换为 OpenCV 的矩阵格式
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            // 计算分数矩阵中的最大值和其索引
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            // 如果最大分数超过矩形置信度阈值，则记录相关信息
            if (maxClassScore > rectConfidenceThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                // 计算边界框的左上角坐标及其宽度和高度
                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);
                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                // 将计算得到的边界框添加到 boxes 向量中
                boxes.push_back(cv::Rect(left, top, width, height));
            }
            // 移动到下一行的数据
            data += signalResultNum;
        }
        // 进行非极大值抑制（NMS）处理，剔除重叠的边界框
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        // 将符合条件的检测结果转换为自定义的结果结构，并添加到最终结果向量中
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }
#ifdef benchmark
        // 如果定义了 benchmark 宏，则进行性能统计
        clock_t starttime_4 = clock();
        // 计算预处理时间
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        // 计算推理时间
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        // 计算后处理时间
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        // 根据是否启用 CUDA 输出不同的性能统计信息
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
        else
        {
            std::cout << "[YOLO_V8(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
        }
#endif // benchmark

        break;
    }
    case YOLO_CLS:
    case YOLO_CLS_HALF:
    {
        cv::Mat rawData;
        if (modelType == YOLO_CLS) {
            // 使用单通道 CV_32F 类型的数据填充 rawData
            rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
        } else {
            // 使用单通道 CV_16F 类型的数据填充 rawData，并将其转换为 CV_32F 类型
            rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++)
        {
            // 设置结果中的类别 ID 和置信度
            result.classId = i;
            result.confidence = data[i];
            // 将结果添加到输出结果列表中
            oResult.push_back(result);
        }
        break;
    }
    default:
        // 如果模型类型不支持，则输出错误信息
        std::cout << "[YOLO_V8]: " << "Not support model type." << std::endl;
    }
    // 返回函数执行结果 OK
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    // 记录开始时间点
    clock_t starttime_1 = clock();
    // 创建指定大小的空白图像 iImg
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    // 对输入图像进行预处理
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        // 如果模型类型小于 4，即使用 FP32 或 FP16 进行推理
        float* blob = new float[iImg.total() * 3];
        // 从处理后的图像生成 blob
        BlobFromImage(processedImg, blob);
        // 设置输入节点的维度信息
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        // 创建输入张量
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        // 运行推理会话并获取输出张量
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
            outputNodeNames.size());
        delete[] blob;
        // 计算预处理到后处理的时间
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        // 如果启用了 CUDA，则输出 CUDA 加速的性能信息
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    }
    else
    {
#ifdef USE_CUDA
        // 分配存储空间用于存储处理后图像的半精度数据
        half* blob = new half[iImg.total() * 3];
        // 将处理后的图像转换为半精度数据存储在blob中
        BlobFromImage(processedImg, blob);
        // 定义YOLO模型输入节点的维度
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        // 创建输入张量，包含半精度数据和相关维度信息
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        // 运行推理会话，将输入张量传递给模型，并获取输出张量
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        // 释放占用的存储空间，删除半精度数据数组blob
        delete[] blob;
        // 计算后处理时间，单位毫秒
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        // 如果CUDA启用，则输出CUDA预热所需时间
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    // 返回函数执行成功状态码
    return RET_OK;
}
```