# `.\pytorch\aten\src\ATen\native\vulkan\api\ShaderRegistry.cpp`

```
namespace at {
namespace native {
namespace vulkan {
namespace api {

// 检查是否存在指定名称的着色器
bool ShaderRegistry::has_shader(const std::string& shader_name) {
  // 在着色器列表中查找指定名称的着色器
  const ShaderListing::const_iterator it = listings_.find(shader_name);
  // 返回是否找到该着色器
  return it != listings_.end();
}

// 检查是否存在指定操作名称的调度项
bool ShaderRegistry::has_dispatch(const std::string& op_name) {
  // 在注册表中查找指定操作名称的调度项
  const Registry::const_iterator it = registry_.find(op_name);
  // 返回是否找到该调度项
  return it != registry_.end();
}

// 注册一个着色器
void ShaderRegistry::register_shader(ShaderInfo&& shader_info) {
  // 如果已经存在同名着色器，则抛出异常
  if (has_shader(shader_info.kernel_name)) {
    VK_THROW(
        "Shader with name ", shader_info.kernel_name, "already registered");
  }
  // 否则将新的着色器信息加入到列表中
  listings_.emplace(shader_info.kernel_name, shader_info);
}

// 注册操作调度项
void ShaderRegistry::register_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  // 如果操作名称不存在，则在注册表中添加新的调度项
  if (!has_dispatch(op_name)) {
    registry_.emplace(op_name, Dispatcher());
  }
  // 查找指定调度键对应的调度项
  const Dispatcher::const_iterator it = registry_[op_name].find(key);
  // 如果已经存在该调度项，则更新其对应的着色器名称
  if (it != registry_[op_name].end()) {
    registry_[op_name][key] = shader_name;
  } else {
    // 否则添加新的调度项及其对应的着色器名称
    registry_[op_name].emplace(key, shader_name);
  }
}

// 获取指定着色器名称的着色器信息
const ShaderInfo& ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  // 在着色器列表中查找指定名称的着色器信息
  const ShaderListing::const_iterator it = listings_.find(shader_name);

  // 如果找不到则抛出异常
  VK_CHECK_COND(
      it != listings_.end(),
      "Could not find ShaderInfo with name ",
      shader_name);

  // 返回找到的着色器信息
  return it->second;
}

// 返回静态的着色器注册表对象
ShaderRegistry& shader_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```