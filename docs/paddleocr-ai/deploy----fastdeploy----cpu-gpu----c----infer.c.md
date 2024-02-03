# `.\PaddleOCR\deploy\fastdeploy\cpu-gpu\c\infer.c`

```py
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本授权
// 只有在遵守许可证的情况下才能使用此文件
// 您可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 基于“按原样”分发，没有任何明示或暗示的保证或条件
// 请查看许可证以获取有关权限和限制的具体语言

#include <stdio.h>
#include <stdlib.h>

#include "fastdeploy_capi/vision.h"

#ifdef WIN32
// 如果是 Windows 系统，使用反斜杠作为路径分隔符
const char sep = '\\';
#else
// 如果不是 Windows 系统，使用正斜杠作为路径分隔符
const char sep = '/';
#endif
// 执行 CPU 推断，传入检测模型目录、分类模型目录、识别模型目录、识别标签文件和图像文件
void CpuInfer(const char *det_model_dir, const char *cls_model_dir,
              const char *rec_model_dir, const char *rec_label_file,
              const char *image_file) {
  // 定义检测模型文件和参数文件的字符数组
  char det_model_file[100];
  char det_params_file[100];

  // 定义分类模型文件和参数文件的字符数组
  char cls_model_file[100];
  char cls_params_file[100];

  // 定义识别模型文件和参数文件的字符数组
  char rec_model_file[100];
  char rec_params_file[100];

  // 定义最大大小为 99
  int max_size = 99;
  // 格式化生成检测模型文件和参数文件的路径
  snprintf(det_model_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdmodel");
  snprintf(det_params_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdiparams");

  // 格式化生成分类模型文件和参数文件的路径
  snprintf(cls_model_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdmodel");
  snprintf(cls_params_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdiparams");

  // 格式化生成识别模型文件和参数文件的路径
  snprintf(rec_model_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdmodel");
  snprintf(rec_params_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdiparams");

  // 创建检测模型运行时选项
  FD_C_RuntimeOptionWrapper *det_option = FD_C_CreateRuntimeOptionWrapper();
  // 创建分类模型运行时选项
  FD_C_RuntimeOptionWrapper *cls_option = FD_C_CreateRuntimeOptionWrapper();
  // 创建识别模型运行时选项
  FD_C_RuntimeOptionWrapper *rec_option = FD_C_CreateRuntimeOptionWrapper();
  // 使用 CPU 运行时选项
  FD_C_RuntimeOptionWrapperUseCpu(det_option);
  FD_C_RuntimeOptionWrapperUseCpu(cls_option);
  FD_C_RuntimeOptionWrapperUseCpu(rec_option);

  // 创建检测模型包装器
  FD_C_DBDetectorWrapper *det_model = FD_C_CreateDBDetectorWrapper(
      det_model_file, det_params_file, det_option, FD_C_ModelFormat_PADDLE);
  // 创建分类模型包装器
  FD_C_ClassifierWrapper *cls_model = FD_C_CreateClassifierWrapper(
      cls_model_file, cls_params_file, cls_option, FD_C_ModelFormat_PADDLE);
  // 创建识别模型包装器
  FD_C_RecognizerWrapper *rec_model = FD_C_CreateRecognizerWrapper(
      rec_model_file, rec_params_file, rec_label_file, rec_option,
      FD_C_ModelFormat_PADDLE);

  // 创建 PPOCRv3 包装器
  FD_C_PPOCRv3Wrapper *ppocr_v3 =
      FD_C_CreatePPOCRv3Wrapper(det_model, cls_model, rec_model);
  // 如果 PPOCRv3 包装器未初始化
  if (!FD_C_PPOCRv3WrapperInitialized(ppocr_v3)) {
  // 打印初始化失败信息
  printf("Failed to initialize.\n");
  // 销毁运行时选项包装器
  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  // 销毁分类器模型和检测器模型
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  // 销毁识别器模型
  FD_C_DestroyRecognizerWrapper(rec_model);
  // 销毁文本识别模型
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  // 返回
  return;
}

// 读取图像文件并创建图像对象
FD_C_Mat im = FD_C_Imread(image_file);

// 分配内存以存储OCR结果
FD_C_OCRResult *result = (FD_C_OCRResult *)malloc(sizeof(FD_C_OCRResult));

// 使用文本识别模型预测OCR结果
if (!FD_C_PPOCRv3WrapperPredict(ppocr_v3, im, result)) {
  // 打印预测失败信息
  printf("Failed to predict.\n");
  // 销毁运行时选项包装器
  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  // 销毁分类器模型和检测器模型
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  // 销毁识别器模型
  FD_C_DestroyRecognizerWrapper(rec_model);
  // 销毁文本识别模型、图像对象和结果对象
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  FD_C_DestroyMat(im);
  free(result);
  // 返回
  return;
}

// 打印OCR结果
char res[2000];
FD_C_OCRResultStr(result, res);
printf("%s", res);

// 可视化OCR结果并保存为图像文件
FD_C_Mat vis_im = FD_C_VisOcr(im, result);
FD_C_Imwrite("vis_result.jpg", vis_im);
printf("Visualized result saved in ./vis_result.jpg\n");

// 销毁所有模型、选项和对象
FD_C_DestroyRuntimeOptionWrapper(det_option);
FD_C_DestroyRuntimeOptionWrapper(cls_option);
FD_C_DestroyRuntimeOptionWrapper(rec_option);
FD_C_DestroyClassifierWrapper(cls_model);
FD_C_DestroyDBDetectorWrapper(det_model);
FD_C_DestroyRecognizerWrapper(rec_model);
FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
FD_C_DestroyOCRResult(result);
FD_C_DestroyMat(im);
FD_C_DestroyMat(vis_im);
// 执行 GPU 推断，传入检测模型目录、分类模型目录、识别模型目录、识别标签文件和图像文件
void GpuInfer(const char *det_model_dir, const char *cls_model_dir,
              const char *rec_model_dir, const char *rec_label_file,
              const char *image_file) {
  // 定义检测模型文件和参数文件的字符数组
  char det_model_file[100];
  char det_params_file[100];

  // 定义分类模型文件和参数文件的字符数组
  char cls_model_file[100];
  char cls_params_file[100];

  // 定义识别模型文件和参数文件的字符数组
  char rec_model_file[100];
  char rec_params_file[100];

  // 定义最大大小为 99
  int max_size = 99;
  // 格式化生成检测模型文件和参数文件的路径
  snprintf(det_model_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdmodel");
  snprintf(det_params_file, max_size, "%s%c%s", det_model_dir, sep,
           "inference.pdiparams");

  // 格式化生成分类模型文件和参数文件的路径
  snprintf(cls_model_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdmodel");
  snprintf(cls_params_file, max_size, "%s%c%s", cls_model_dir, sep,
           "inference.pdiparams");

  // 格式化生成识别模型文件和参数文件的路径
  snprintf(rec_model_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdmodel");
  snprintf(rec_params_file, max_size, "%s%c%s", rec_model_dir, sep,
           "inference.pdiparams");

  // 创建检测模型运行时选项
  FD_C_RuntimeOptionWrapper *det_option = FD_C_CreateRuntimeOptionWrapper();
  // 创建分类模型运行时选项
  FD_C_RuntimeOptionWrapper *cls_option = FD_C_CreateRuntimeOptionWrapper();
  // 创建识别模型运行时选项
  FD_C_RuntimeOptionWrapper *rec_option = FD_C_CreateRuntimeOptionWrapper();
  // 设置不使用 GPU 运行
  FD_C_RuntimeOptionWrapperUseGpu(det_option, 0);
  FD_C_RuntimeOptionWrapperUseGpu(cls_option, 0);
  FD_C_RuntimeOptionWrapperUseGpu(rec_option, 0);

  // 创建检测模型包装器
  FD_C_DBDetectorWrapper *det_model = FD_C_CreateDBDetectorWrapper(
      det_model_file, det_params_file, det_option, FD_C_ModelFormat_PADDLE);
  // 创建分类模型包装器
  FD_C_ClassifierWrapper *cls_model = FD_C_CreateClassifierWrapper(
      cls_model_file, cls_params_file, cls_option, FD_C_ModelFormat_PADDLE);
  // 创建识别模型包装器
  FD_C_RecognizerWrapper *rec_model = FD_C_CreateRecognizerWrapper(
      rec_model_file, rec_params_file, rec_label_file, rec_option,
      FD_C_ModelFormat_PADDLE);

  // 创建 PPOCRv3 包装器
  FD_C_PPOCRv3Wrapper *ppocr_v3 =
      FD_C_CreatePPOCRv3Wrapper(det_model, cls_model, rec_model);
  // 如果 PPOCRv3 包装器未初始化
  if (!FD_C_PPOCRv3WrapperInitialized(ppocr_v3)) {
  // 打印初始化失败信息
  printf("Failed to initialize.\n");
  // 销毁运行时选项包装器
  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  // 销毁分类器模型和检测器模型
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  // 销毁识别器模型
  FD_C_DestroyRecognizerWrapper(rec_model);
  // 销毁文本识别模型
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  // 返回
  return;
}

// 读取图像文件并创建图像对象
FD_C_Mat im = FD_C_Imread(image_file);

// 分配内存以存储OCR结果
FD_C_OCRResult *result = (FD_C_OCRResult *)malloc(sizeof(FD_C_OCRResult));

// 使用文本识别模型预测OCR结果
if (!FD_C_PPOCRv3WrapperPredict(ppocr_v3, im, result)) {
  // 打印预测失败信息
  printf("Failed to predict.\n");
  // 销毁运行时选项包装器
  FD_C_DestroyRuntimeOptionWrapper(det_option);
  FD_C_DestroyRuntimeOptionWrapper(cls_option);
  FD_C_DestroyRuntimeOptionWrapper(rec_option);
  // 销毁分类器模型和检测器模型
  FD_C_DestroyClassifierWrapper(cls_model);
  FD_C_DestroyDBDetectorWrapper(det_model);
  // 销毁识别器模型
  FD_C_DestroyRecognizerWrapper(rec_model);
  // 销毁文本识别模型、图像对象和结果对象
  FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
  FD_C_DestroyMat(im);
  free(result);
  // 返回
  return;
}

// 打印OCR结果
char res[2000];
FD_C_OCRResultStr(result, res);
printf("%s", res);

// 可视化OCR结果并保存为图像文件
FD_C_Mat vis_im = FD_C_VisOcr(im, result);
FD_C_Imwrite("vis_result.jpg", vis_im);
printf("Visualized result saved in ./vis_result.jpg\n");

// 销毁所有模型、选项和对象
FD_C_DestroyRuntimeOptionWrapper(det_option);
FD_C_DestroyRuntimeOptionWrapper(cls_option);
FD_C_DestroyRuntimeOptionWrapper(rec_option);
FD_C_DestroyClassifierWrapper(cls_model);
FD_C_DestroyDBDetectorWrapper(det_model);
FD_C_DestroyRecognizerWrapper(rec_model);
FD_C_DestroyPPOCRv3Wrapper(ppocr_v3);
FD_C_DestroyOCRResult(result);
FD_C_DestroyMat(im);
FD_C_DestroyMat(vis_im);
// 主函数，接受命令行参数
int main(int argc, char *argv[]) {
  // 检查参数数量是否小于7，如果小于7则输出提示信息并返回-1
  if (argc < 7) {
    printf("Usage: infer_demo path/to/det_model path/to/cls_model "
           "path/to/rec_model path/to/rec_label_file path/to/image "
           "run_option, "
           "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
           "./ch_ppocr_mobile_v3.0_cls_infer ./ch_PP-OCRv3_rec_infer "
           "./ppocr_keys_v1.txt ./12.jpg 0\n");
    printf(
        "The data type of run_option is int, 0: run with cpu; 1: run with gpu"
        "\n");
    return -1;
  }

  // 检查第6个参数是否为0，如果是则调用CpuInfer函数
  if (atoi(argv[6]) == 0) {
    CpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } 
  // 如果第6个参数不为0，则检查是否为1，如果是则调用GpuInfer函数
  else if (atoi(argv[6]) == 1) {
    GpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  }
  // 返回0表示程序正常结束
  return 0;
}
```