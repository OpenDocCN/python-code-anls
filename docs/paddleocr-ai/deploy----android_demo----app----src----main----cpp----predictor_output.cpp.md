# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\predictor_output.cpp`

```
namespace ppredictor {
// 返回浮点数据的指针
const float *PredictorOutput::get_float_data() const {
  return _tensor->data<float>();
}

// 返回整型数据的指针
const int *PredictorOutput::get_int_data() const {
  return _tensor->data<int>();
}

// 返回数据的级联偏移描述
const std::vector<std::vector<uint64_t>> PredictorOutput::get_lod() const {
  return _tensor->lod();
}

// 返回数据的大小
int64_t PredictorOutput::get_size() const {
  // 如果网络标志为 NET_OCR，则返回第二维和第三维的乘积
  if (_net_flag == NET_OCR) {
    return _tensor->shape().at(2) * _tensor->shape().at(3);
  } else {
    // 否则返回数据形状的乘积
    return product(_tensor->shape());
  }
}

// 返回数据的形状
const std::vector<int64_t> PredictorOutput::get_shape() const {
  return _tensor->shape();
}
}
```