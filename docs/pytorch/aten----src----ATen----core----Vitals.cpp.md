# `.\pytorch\aten\src\ATen\core\Vitals.cpp`

```
namespace at::vitals {

命名空间 `at::vitals` 的开始。


APIVitals VitalsAPI;

定义了一个全局变量 `VitalsAPI`，类型为 `APIVitals`，用于管理与 API 相关的关键信息。


std::ostream& operator<<(std::ostream& os, TorchVital const& tv) {

重载了输出流操作符 `<<`，使其能够打印 `TorchVital` 类型对象到输出流 `os` 中。


for (const auto& m : tv.attrs) {

对 `TorchVital` 对象 `tv` 中的属性 `attrs` 进行循环迭代。


os << "[TORCH_VITAL] " << tv.name << "." << m.first << "\t\t "
   << m.second.value << "\n";

将 `tv` 对象的名称 `tv.name`、属性名 `m.first` 和属性值 `m.second.value` 输出到流 `os` 中，格式为 "[TORCH_VITAL] 名称.属性名     属性值\n"。


return os;

返回输出流 `os`，完成对 `TorchVital` 对象的输出。


TorchVital::~TorchVital() {

`TorchVital` 类的析构函数的开始。


if (torchVitalEnabled()) {

如果 `torchVitalEnabled()` 函数返回 true，即 Torch Vital 功能已启用。


std::cout << *this;

输出当前 `TorchVital` 对象的信息到标准输出流。


}

结束条件判断块。


}

析构函数的结束。


TorchVitalAttr& TorchVital::create(const std::string& attr) {

`TorchVital` 类的成员函数 `create` 的声明，用于创建指定属性名称的 `TorchVitalAttr` 对象的引用。


return create(attr, /* force = */ false);

调用重载的 `create` 函数，传入属性名称 `attr` 和 `force` 参数为 false。


TorchVitalAttr& TorchVital::create(const std::string& attr, bool force) {

重载的 `create` 函数的开始，用于根据指定属性名称 `attr` 创建 `TorchVitalAttr` 对象的引用，可选地强制创建。


if (!(torchVitalEnabled() || force)) {

如果 Torch Vital 功能未启用且未强制创建，则返回一个静态的禁用属性对象 `disabled`。


auto iter = attrs.find(attr);

查找属性名称 `attr` 是否已存在于 `attrs` 映射中。


if (iter == attrs.end()) {

如果属性名称 `attr` 不存在于 `attrs` 映射中。


auto r = attrs.emplace(attr, TorchVitalAttr());

在 `attrs` 映射中插入新的属性 `attr`，对应一个新的 `TorchVitalAttr` 对象，并返回插入结果 `r`。


return r.first->second;

返回新插入的属性 `attr` 对应的 `TorchVitalAttr` 对象的引用。


}

结束条件判断块。


return iter->second;

返回已存在的属性 `attr` 对应的 `TorchVitalAttr` 对象的引用。


}

`create` 函数的结束。


bool torchVitalEnabled() {

`torchVitalEnabled` 函数的开始，用于检查 Torch Vital 是否启用。


bool enabled = []() {

定义了一个 lambda 函数，返回当前环境中是否设置了 `TORCH_VITAL` 环境变量来判断是否启用 Torch Vital 功能。


if (e != nullptr) {

如果环境变量 `e` 不为 nullptr。


return e[0] != '\0';

返回 `TORCH_VITAL` 环境变量的第一个字符是否不为 '\0'，即环境变量是否非空字符串。


}

结束条件判断块。


return false;

如果环境变量 `TORCH_VITAL` 未设置或为空字符串，则返回 false。


}();

调用定义的 lambda 函数并将结果赋给 `enabled` 变量。


if (enabled) {

如果 Torch Vital 功能已启用。


VitalsAPI.vitals_enabled = true;

设置全局变量 `VitalsAPI` 中的 `vitals_enabled` 为 true。


}

结束条件判断块。


return VitalsAPI.vitals_enabled;

返回 Torch Vital 功能的启用状态。


}

`torchVitalEnabled` 函数的结束。


std::string APIVitals::readVitals() {

`APIVitals` 类的成员函数 `readVitals` 的开始，用于读取所有已记录的 Vital 数据并返回字符串形式。


if (!torchVitalEnabled()) {

如果 Torch Vital 功能未启用。


return "";

直接返回空字符串。


}

结束条件判断块。


std::stringstream buf;

创建一个字符串流 `buf`，用于存储所有 Vital 数据的字符串形式。


for (const auto& x : name_map_) {

对 `name_map_` 中的每个元素 `x` 进行循环迭代。


buf << x.second;

将当前 `x` 对应的 `TorchVital` 对象的字符串形式追加到字符串流 `buf` 中。


}

结束循环。


return buf.str();

返回字符串流 `buf` 中的所有数据作为一个字符串。


}

`readVitals` 函数的结束。


bool APIVitals::setVital(
    const std::string& vital_name,
    const std::string& attr_name,
    const std::string& value,
    bool force) {

`APIVitals` 类的成员函数 `setVital` 的开始，用于设置指定 Vital 的指定属性和值。


if (!(torchVitalEnabled() || force)) {

如果 Torch Vital 功能未启用且未强制设置，则返回 false。


auto iter = name_map_.find(vital_name);

查找名称为 `vital_name` 的 Vital 是否已存在于 `name_map_` 中。


TorchVital* vital = nullptr;

声明指向 `TorchVital` 对象的指针 `vital`，初始化为 nullptr。


if (iter == name_map_.end()) {

如果名称为 `vital_name` 的 Vital 不存在于 `name_map_` 中。


auto r = name_map_.emplace(vital_name, TorchVital(vital_name));

在 `name_map_` 中插入新的 Vital，对应一个新创建的 `TorchVital` 对象，并返回插入结果 `r`。


vital = &r.first->second;

将 `vital` 指针指向新插入的 `TorchVital` 对象。


} else {

否则，如果名称为 `vital_name` 的 Vital 已存在于 `name_map_` 中。


vital = &iter->second;

将 `vital` 指针指向已存在的 `TorchVital` 对象。


}

结束条件判断块。


vital->create(attr_name, force).write(value, force);

调用 `vital` 指针指向的 `TorchVital` 对象的 `create` 函数创建指定属性 `attr_name`，并调用其 `write` 函数写入指定值 `value`，可选地强制写入。


return true;

返回设置成功。


}

`setVital` 函数的结束。


APIVitals::APIVitals() : vitals_enabled(false), name_map_() {

`APIVitals` 类的构造函数的开始，初始化 `vitals_enabled` 为 false，`name_map_` 为空。


setVital("CUDA", "used", "False", /* force = */ true);

调用 `setVital` 函数，设置名称为 "CUDA" 的 Vital 的 "used" 属性为 "False"，并强制设置。


}

`APIVitals` 类的构造函数的结束。


}
```