# `.\pytorch\aten\src\ATen\nnapi\nnapi_model_loader.cpp`

```
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdint.h>  // 包含用于整数类型的标准头文件

#include <ATen/nnapi/NeuralNetworks.h>  // 包含 ATen 的 NNAPI 相关头文件
#include <ATen/nnapi/nnapi_wrapper.h>
#include <ATen/nnapi/nnapi_model_loader.h>
#include <c10/util/irange.h>  // 包含 c10 库的 irange.h 头文件

#ifndef NNAPI_LOADER_STANDALONE
# include <c10/util/Logging.h>  // 如果不是独立模式，包含 c10 的 Logging 头文件
#else
# define CAFFE_ENFORCE(cond, ...) do { if (!cond) { return -1; } } while (0)  // 在独立模式下定义 CAFFE_ENFORCE 宏
#endif

#define NNAPI_CHECK(res) CAFFE_ENFORCE(res == ANEURALNETWORKS_NO_ERROR, "NNAPI returned error: ", res)  // 定义 NNAPI_CHECK 宏，检查返回结果是否为错误

namespace caffe2 {
namespace nnapi {

namespace {

/*
Serialized format for NNAPI models.  It is basically just a list arguments
for calls to be made to NNAPI.
*/
// NNAPI 模型的序列化格式，基本上只是调用 NNAPI 所需的参数列表

typedef enum _SourceType {
  SOURCE_IMMEDIATE = 0,  // 立即类型的数据源
  SOURCE_NUMBERED_BUFFER = 2,  // 编号缓冲区类型的数据源
  SOURCE_NUMBERED_MEMORY = 3,  // 编号内存类型的数据源
} SourceType;  // 定义数据源类型枚举

typedef struct _SerializedOperand {
  int32_t type;  // 操作数类型
  uint32_t dimension_count;  // 维度数量
  float scale;  // 比例
  int32_t zero_point;  // 零点
} SerializedOperand;  // 序列化操作数结构体

typedef struct _SerializedValue {
  int32_t index;  // 索引
  int32_t source_type;  // 数据源类型
  uint32_t source_length;  // 数据源长度
} SerializedValue;  // 序列化数值结构体

typedef struct _SerializedOperation {
  int32_t operation_type;  // 操作类型
  uint32_t input_count;  // 输入数量
  uint32_t output_count;  // 输出数量
} SerializedOperation;  // 序列化操作结构体

typedef struct _SerializedModel {
  int32_t version;  // 版本
  int32_t operand_count;  // 操作数数量
  int32_t value_count;  // 值数量
  int32_t operation_count;  // 操作数量
  int32_t input_count;  // 输入数量
  int32_t output_count;  // 输出数量
  // SerializedOperand operands[operand_count];  // 操作数数组
  // SerializedValue values[value_count];  // 值数组
  // SerializedOperation operations[operation_count];  // 操作数组
  // uint32_t operand_dimensions[sum(dimension_count)]  // 操作数维度数组
  // uint32_t value_data[sum(source_length+pad)/4]  // 值数据数组
  // uint32_t operation_args[sum(input_count + output_count)]  // 操作参数数组
  // uint32_t model_inputs[input_count]  // 模型输入数组
  // uint32_t model_outputs[output_count]  // 模型输出数组
} SerializedModel;  // 序列化模型结构体

/**
 * Get the physically stored size of a value.  All values are padded out
 * to a multiple of 4 bytes to ensure the next value is 4-byte aligned.
 */
// 获取值的物理存储大小，所有的值都会填充到4字节的倍数，以确保下一个值是4字节对齐的

static uint32_t value_physical_size(uint32_t len) {
  uint32_t phys = len;  // 物理长度等于给定长度
  if (len % 4 == 0) {  // 如果长度是4的倍数
    return len;  // 直接返回长度
  }
  return len + 4 - (phys % 4);  // 否则返回填充到4字节倍数后的长度
}

} // namespace

int load_nnapi_model(
    struct nnapi_wrapper* nnapi,  // NNAPI 包装器指针
    ANeuralNetworksModel* model,  // NNAPI 模型指针
    const void* serialized_model,  // 序列化模型数据指针
    int64_t model_length,  // 模型长度
    size_t num_buffers,  // 缓冲区数量
    const void** buffer_ptrs,  // 缓冲区指针数组
    int32_t* buffer_sizes,  // 缓冲区大小数组
    size_t /*num_memories*/,  // 内存数量（未使用）
    ANeuralNetworksMemory** /*memories*/,  // NNAPI 内存指针数组（未使用）
    int32_t* /*memory_sizes*/,  // 内存大小数组（未使用）
    int32_t* out_input_count,  // 输出的输入数量
    int32_t* out_output_count,  // 输出的输出数量
    size_t* out_bytes_consumed) {
  // 计算所需的模型序列化数据的总大小
  int64_t required_size = 0;
  // 指向序列化模型数据的下一个字节指针
  const uint8_t* next_pointer = (const uint8_t*)serialized_model;
  // 指向序列化模型数据末尾的指针
  const uint8_t* end_of_buf = (const uint8_t*)serialized_model + model_length;

  // 添加 SerializedModel 结构体的大小到所需大小
  required_size += sizeof(SerializedModel);
  // 确保模型数据长度不小于所需大小，否则抛出异常
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  // 将指针转换为 SerializedModel 类型
  const SerializedModel* ser_model = (SerializedModel*)next_pointer;
  // 更新下一个指针位置
  next_pointer = (uint8_t*)serialized_model + required_size;
  // 确保下一个指针位置不超过模型数据的末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 确保 SerializedModel 结构体的版本号为 1
  CAFFE_ENFORCE(ser_model->version == 1);
  // 检查一些值是否小于 2^24，以避免整数溢出
  CAFFE_ENFORCE(ser_model->operand_count    < (1 << 24));
  CAFFE_ENFORCE(ser_model->value_count      < (1 << 24));
  CAFFE_ENFORCE(ser_model->operation_count  < (1 << 24));
  CAFFE_ENFORCE(ser_model->input_count      < (1 << 24));
  CAFFE_ENFORCE(ser_model->output_count     < (1 << 24));

  // 添加 SerializedOperand 结构体数组的大小到所需大小
  required_size += sizeof(SerializedOperand) * ser_model->operand_count;
  // 确保模型数据长度不小于所需大小，否则抛出异常
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  // 将指针转换为 SerializedOperand 类型数组
  const SerializedOperand* operands = (const SerializedOperand*)next_pointer;
  // 更新下一个指针位置
  next_pointer = (uint8_t*)serialized_model + required_size;
  // 确保下一个指针位置不超过模型数据的末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 添加 SerializedValue 结构体数组的大小到所需大小
  required_size += sizeof(SerializedValue) * ser_model->value_count;
  // 确保模型数据长度不小于所需大小，否则抛出异常
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  // 将指针转换为 SerializedValue 类型数组
  const SerializedValue* values = (const SerializedValue*)next_pointer;
  // 更新下一个指针位置
  next_pointer = (uint8_t*)serialized_model + required_size;
  // 确保下一个指针位置不超过模型数据的末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 添加 SerializedOperation 结构体数组的大小到所需大小
  required_size += sizeof(SerializedOperation) * ser_model->operation_count;
  // 确保模型数据长度不小于所需大小，否则抛出异常
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  // 将指针转换为 SerializedOperation 类型数组
  const SerializedOperation* operations = (const SerializedOperation*)next_pointer;
  // 更新下一个指针位置
  next_pointer = (uint8_t*)serialized_model + required_size;
  // 确保下一个指针位置不超过模型数据的末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 遍历模型中的每个操作数，计算所需的内存大小
  for (const auto i : c10::irange(ser_model->operand_count)) {
    required_size += 4 * operands[i].dimension_count;
  }

  // 遍历模型中的每个值，计算所需的内存大小
  for (const auto i : c10::irange(ser_model->value_count)) {
    required_size += value_physical_size(values[i].source_length);
  }

  // 遍历模型中的每个操作，计算所需的内存大小
  for (const auto i : c10::irange(ser_model->operation_count)) {
    required_size += 4 * (operations[i].input_count + operations[i].output_count);
  }

  // 计算输入和输出张量的总大小
  required_size += 4 * (ser_model->input_count + ser_model->output_count);

  // 确保模型数据长度不小于最终所需大小，否则抛出异常
  CAFFE_ENFORCE(model_length >= required_size, "Model is too small.  Size = ", model_length);
  // 确保下一个指针位置不超过模型数据的末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 将模型中的每个操作数的信息填充到 ANeuralNetworksOperandType 结构体中
  for (const auto i : c10::irange(ser_model->operand_count)) {
    ANeuralNetworksOperandType operand;
    operand.type = operands[i].type;
    operand.scale = operands[i].scale;
    operand.zeroPoint = operands[i].zero_point;
    operand.dimensionCount = operands[i].dimension_count;
    operand.dimensions = operands[i].dimension_count ? (const uint32_t*)next_pointer : nullptr;
    // 计算下一个指针位置，移动到当前操作数的下一个操作数的起始位置
    next_pointer += 4 * operands[i].dimension_count;
    // 确保下一个指针位置不超过缓冲区的结束位置
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    // 向 NNAPI 模型中添加操作数，并检查操作结果
    int result = nnapi->Model_addOperand(model, &operand);
    NNAPI_CHECK(result);
  }

  // 遍历序列化模型中的值
  for (const auto i : c10::irange(ser_model->value_count)) {
    // 获取当前值的长度
    uint32_t len = values[i].source_length;
    // 存储当前指针位置到 stored_pointer
    const uint8_t* stored_pointer = next_pointer;
    const void* value_pointer = nullptr;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t value_length;

    // 根据值的来源类型执行不同的处理
    switch ((SourceType)values[i].source_type) {
      case SOURCE_IMMEDIATE:
        {
          // 对于立即数，直接使用当前存储指针作为值指针，长度为 len
          value_pointer = stored_pointer;
          value_length = len;
        }
        break;
      case SOURCE_NUMBERED_BUFFER:
        {
          // 对于编号缓冲区，从存储指针中解析出缓冲区编号、偏移和长度信息
          CAFFE_ENFORCE(len == 12);
          uint32_t buffer_number = *(uint32_t*)stored_pointer;
          uint32_t buffer_offset = *(uint32_t*)(stored_pointer + 4);
          uint32_t operand_length = *(uint32_t*)(stored_pointer + 8);
          // 确保缓冲区编号在有效范围内
          CAFFE_ENFORCE(buffer_number < num_buffers);
          // 确保操作不会导致缓冲区溢出
          CAFFE_ENFORCE(buffer_offset + operand_length >= buffer_offset);  // No integer overflow
          CAFFE_ENFORCE(buffer_offset + operand_length <= (uint32_t)buffer_sizes[buffer_number]);  // No buffer overflow
          // 设置值指针为缓冲区起始位置加上偏移量
          value_pointer = (uint8_t*)buffer_ptrs[buffer_number] + buffer_offset;
          value_length = operand_length;
        }
        break;
      case SOURCE_NUMBERED_MEMORY:
        // 对于编号内存，暂时不支持
        CAFFE_ENFORCE(false, "Memory inputs not implemented yet.");
        break;
      default:
        // 对于未知的值来源类型，抛出错误
        CAFFE_ENFORCE(false, "Unknown source type: ", values[i].source_type);
    }

    // 确保值指针不为空
    CAFFE_ENFORCE(value_pointer != nullptr);

    // 更新下一个指针位置，移动到当前值的下一个物理大小的位置
    next_pointer += value_physical_size(len);
    // 确保下一个指针位置不超过缓冲区的结束位置
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    // 向 NNAPI 模型设置操作数的值，并检查操作结果
    int result = nnapi->Model_setOperandValue(
        model,
        values[i].index,
        value_pointer,
        value_length);
    NNAPI_CHECK(result);
  }

  // 遍历序列化模型中的操作
  for (const auto i : c10::irange(ser_model->operation_count)) {
    // 获取当前操作的输入指针
    const uint32_t* inputs = (const uint32_t*)next_pointer;
    // 更新下一个指针位置，移动到当前操作的输出指针起始位置
    next_pointer += 4 * operations[i].input_count;
    // 确保下一个指针位置不超过缓冲区的结束位置
    CAFFE_ENFORCE(next_pointer <= end_of_buf);
    // 获取当前操作的输出指针
    const uint32_t* outputs = (const uint32_t*)next_pointer;
    // 更新下一个指针位置，移动到下一个操作的起始位置
    next_pointer += 4 * operations[i].output_count;
    // 确保下一个指针位置不超过缓冲区的结束位置
    CAFFE_ENFORCE(next_pointer <= end_of_buf);

    // 向 NNAPI 模型添加操作，并检查操作结果
    int result = nnapi->Model_addOperation(
        model,
        operations[i].operation_type,
        operations[i].input_count,
        inputs,
        operations[i].output_count,
        outputs);
    NNAPI_CHECK(result);
  }

  // 将 next_pointer 解释为模型输入参数的无符号整数数组指针
  const uint32_t* model_inputs = (const uint32_t*)next_pointer;
  // 将 next_pointer 向后移动到模型输入参数数组结束位置
  next_pointer += 4 * ser_model->input_count;
  // 确保 next_pointer 未超出缓冲区末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);
  // 将 next_pointer 解释为模型输出参数的无符号整数数组指针
  const uint32_t* model_outputs = (const uint32_t*)next_pointer;
  // 将 next_pointer 向后移动到模型输出参数数组结束位置
  next_pointer += 4 * ser_model->output_count;
  // 确保 next_pointer 未超出缓冲区末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);

  // 调用 NNAPI 接口识别模型的输入和输出
  int result = nnapi->Model_identifyInputsAndOutputs(
      model,
      ser_model->input_count,
      model_inputs,
      ser_model->output_count,
      model_outputs);
  NNAPI_CHECK(result);

  // 将输出参数写入 out_input_count 和 out_output_count 指向的变量
  *out_input_count = ser_model->input_count;
  *out_output_count = ser_model->output_count;

  // TODO: 可能可以消除 required_size，只依赖 next_pointer 进行边界检查。
  // 确保 next_pointer 未超出缓冲区末尾
  CAFFE_ENFORCE(next_pointer <= end_of_buf);
  // 确保 next_pointer 正好指向 serialized_model 的末尾
  CAFFE_ENFORCE(next_pointer == (const uint8_t*)serialized_model + required_size);
  // 如果 out_bytes_consumed 不为空，则计算已消耗的字节数
  if (out_bytes_consumed != nullptr) {
    *out_bytes_consumed = next_pointer - (const uint8_t*)serialized_model;
  }

  // 函数执行成功返回 0
  return 0;
}

}} // namespace caffe2::nnapi
```