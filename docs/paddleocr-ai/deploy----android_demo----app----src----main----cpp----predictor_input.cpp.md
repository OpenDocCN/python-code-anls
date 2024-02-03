# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\predictor_input.cpp`

```py
namespace ppredictor {

void PredictorInput::set_dims(std::vector<int64_t> dims) {
  // 如果是 yolov3 网络且索引为1，则设置张量大小为{1, 2}，并将维度信息写入张量数据
  if (_net_flag == 101 && _index == 1) {
    _tensor->Resize({1, 2});
    _tensor->mutable_data<int>()[0] = (int)dims.at(2);
    _tensor->mutable_data<int>()[1] = (int)dims.at(3);
  } else {
    // 否则根据给定的维度设置张量大小
    _tensor->Resize(dims);
  }
  // 标记维度已设置
  _is_dims_set = true;
}

float *PredictorInput::get_mutable_float_data() {
  // 如果维度未设置，则输出错误日志
  if (!_is_dims_set) {
    LOGE("PredictorInput::set_dims is not called");
  }
  // 返回可变的浮点数据指针
  return _tensor->mutable_data<float>();
}

void PredictorInput::set_data(const float *input_data, int input_float_len) {
  // 获取可变的浮点数据指针
  float *input_raw_data = get_mutable_float_data();
  // 将输入数据复制到张量数据中
  memcpy(input_raw_data, input_data, input_float_len * sizeof(float));
}
}
```