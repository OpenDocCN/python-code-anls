# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\any_module_holder.h`

```py
    /// 从给定的参数向前调用底层模块的 `forward()` 方法，将每个 `AnyValue` 转换为具体值
    /// 参数:
    /// - arguments: 包含任意类型值的向量，用于调用模块的 `forward()` 方法
    AnyValue forward(std::vector<AnyValue>&& arguments) override {
        // 使用 lambda 函数将 `AnyValue` 转换为具体值并调用模块的 `forward()` 方法
        return apply<ArgumentTypes...>(
            CheckedGetter{arguments},
            InvokeForward{module_});
    }

    /// 返回指向被擦除模块的 `std::shared_ptr<Module>`。
    std::shared_ptr<Module> ptr() override {
        // 返回指向擦除模块的 `std::shared_ptr`
        return module_;
    }

    /// 返回一个浅拷贝的 `AnyModulePlaceholder` 对象，拷贝当前 `AnyModule`
    std::unique_ptr<AnyModulePlaceholder> copy() const override {
        // 返回当前 `AnyModule` 的浅拷贝
        return std::make_unique<AnyModuleHolder<ModuleType, ArgumentTypes...>>(module_);
    }

    /// 返回一个深拷贝的 `AnyModulePlaceholder` 对象，拷贝当前 `AnyModule`
    std::unique_ptr<AnyModulePlaceholder> clone_module(
        optional<Device> device) const override {
        // 返回当前 `AnyModule` 的深拷贝
        return std::make_unique<AnyModuleHolder<ModuleType, ArgumentTypes...>>(module_->clone(device));
    }

  private:
    std::shared_ptr<ModuleType> module_;
};
    // 检查模块是否有默认参数
    if (module->_forward_has_default_args()) {
      // 对于有默认参数的情况，检查传入参数的数量是否在要求的范围内
      TORCH_CHECK(
          arguments.size() >= module->_forward_num_required_args() &&
              arguments.size() <= sizeof...(ArgumentTypes),
          c10::demangle(type_info.name()),
          "'s forward() method expects at least ",
          module->_forward_num_required_args(),
          " argument(s) and at most ",
          sizeof...(ArgumentTypes),
          " argument(s), but received ",
          arguments.size(),
          ".");
      // 使用模块方法填充默认参数，并更新 arguments
      arguments = std::move(
          module->_forward_populate_default_args(std::move(arguments)));
    } else {
      // 对于没有默认参数的情况，检查传入参数数量是否与期望数量相同
      std::string use_default_args_macro_prompt = " If " +
          c10::demangle(type_info.name()) +
          "'s forward() method has default arguments, " +
          "please make sure the forward() method is declared with a corresponding `FORWARD_HAS_DEFAULT_ARGS` macro.";
      TORCH_CHECK(
          arguments.size() == sizeof...(ArgumentTypes),
          c10::demangle(type_info.name()),
          "'s forward() method expects ",
          sizeof...(ArgumentTypes),
          " argument(s), but received ",
          arguments.size(),
          ".",
          (arguments.size() < sizeof...(ArgumentTypes))
              ? use_default_args_macro_prompt  // 如果实际参数少于期望，提醒使用默认参数的宏
              : "");  // 否则为空字符串
    }

    // FYI: 在调用模块的 `forward()` 方法期间，参数值存在于 `arguments` 向量中
    // 使用 `torch::unpack` 解包参数并调用 `forward()` 方法
    return torch::unpack<AnyValue, ArgumentTypes...>(
        InvokeForward{module}, CheckedGetter{arguments});
  }

  // 返回模块的共享指针
  std::shared_ptr<Module> ptr() override {
    return module;
  }

  // 创建并返回当前对象的副本
  std::unique_ptr<AnyModulePlaceholder> copy() const override {
    return std::make_unique<AnyModuleHolder>(*this);
  }

  // 克隆模块到指定设备上，并返回其新的包装对象
  std::unique_ptr<AnyModulePlaceholder> clone_module(
      optional<Device> device) const override {
    return std::make_unique<AnyModuleHolder>(
        std::dynamic_pointer_cast<ModuleType>(module->clone(device)));
  }

  /// 具体的模块实例
  // 指向特定模块实例的智能指针
  std::shared_ptr<ModuleType> module;
};

} // namespace nn
} // namespace torch
```