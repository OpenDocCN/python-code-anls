# `.\pytorch\torch\csrc\jit\mobile\parse_bytecode.cpp`

```
// 包含必要的头文件
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/code.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/mobile/upgrader_mobile.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>
#include <torch/custom_class_detail.h>

// 命名空间声明
namespace torch {
namespace jit {

// 定义解析操作码的函数
OpCode parseOpCode(const char* str);
using c10::IValue;

// 期望解析元组中指定名称的字段
IValue expect_field(
    c10::ivalue::TupleElements& elements,  // 输入的元组元素
    const std::string& expected_name,      // 期望的字段名称
    size_t entry) {                        // 元组中字段的索引位置
  auto row = std::move(elements.at(entry)).toTuple();  // 获取指定索引位置的元组行
  TORCH_INTERNAL_ASSERT(                               // 内部断言，用于检查条件
      row->elements().at(0).toStringRef() == expected_name,  // 检查字段名是否符合预期
      "Expected ", expected_name, " found ", row->elements().at(0).toStringRef());
  return std::move(row)->elements().at(1);  // 返回元组中指定字段的值
}

namespace mobile {

namespace {
#define COUNT_OPCODE(_, _a) 1 +  // 计算所有操作码的数量
constexpr size_t numOpcodes = FORALL_OPCODES(COUNT_OPCODE) 0;  // 实际操作码数量
#undef COUNT_OPCODE

// 操作码缓存类，用于存储操作码的解析结果，加速解析过程
class OpCodeCache {
 private:
  std::array<const void*, numOpcodes> keys_{};    // 存储操作码字符串地址的数组
  std::array<OpCode, numOpcodes> values_{};       // 存储解析后操作码的数组
  size_t usedEntries_ = 0;                        // 当前已使用的条目数量

 public:
  OpCodeCache() {
    memset(keys_.data(), 0, keys_.size() * sizeof(keys_[0]));  // 初始化 keys_ 数组
  }

  // 解析操作码字符串并进行缓存
  OpCode parse(const c10::ivalue::ConstantString& s) {
    const auto endIt = keys_.begin() + usedEntries_;
    auto it = std::find_if(
        keys_.begin(), endIt, [&s](const void* k) { return k == &s; });  // 查找是否已缓存该字符串的操作码
    if (it == endIt) {
      OpCode result = parseOpCode(s.string().c_str());  // 解析操作码字符串
      if (usedEntries_ < numOpcodes) {
        keys_[usedEntries_] = &s;         // 缓存操作码字符串地址
        values_[usedEntries_++] = result; // 缓存解析结果
      }
      return result;  // 返回解析后的操作码
    }
    return values_[it - keys_.begin()];  // 如果已缓存，直接返回缓存的操作码
  }
};
} // namespace

// 应用升级器，更新函数的操作码版本
void applyUpgrader(mobile::Function* function, uint64_t operator_version) {
  Code& code = function->get_code();  // 获取函数的代码对象
  auto& operator_version_map = getOperatorVersionMapForMobile();  // 获取操作码版本映射表
  for (size_t i = 0; i < code.instructions_.size(); i++) {
    Instruction& inst = code.instructions_[i];  // 获取指令对象
    // 如果指令的操作码为 OpCode::OP
    if (inst.op == OpCode::OP) {
      // 获取操作符的名称
      std::string op_name = code.op_names_[inst.X].name;
      // 构建操作符的完整名称，包括重载名称（如果存在）
      std::string operator_name = code.op_names_[inst.X].name +
          (code.op_names_[inst.X].overload_name.empty()
               ? ""
               : "." + code.op_names_[inst.X].overload_name);

      // 在操作符版本映射表中查找该操作符名称
      auto it = operator_version_map.find(operator_name);
      // 检查是否存在该操作符的版本升级器
      if (it != operator_version_map.end()) {
        auto upgrader_list = it->second;
        // 遍历该操作符的所有升级器，查找符合当前操作符版本的有效升级器
        for (const auto& upgrader : upgrader_list) {
          // 如果当前操作符版本在升级器的版本范围内
          if (static_cast<int>(operator_version) <= upgrader.max_version &&
              static_cast<int>(operator_version) >= upgrader.min_version) {
            // 设置指令操作码为 OpCode::CALL，并将索引指向相应的升级器函数
            TORCH_CHECK(
                upgrader.index < static_cast<int>(code.functions_.size()),
                "upgrader index is, ",
                upgrader.index,
                " and it's larger than the upgrader function list length ",
                code.functions_.size());
            inst.op = OpCode::CALL;
            inst.X = upgrader.index;
          }
        }
      }
    }
  }
} // namespace mobile
} // namespace jit
} // namespace torch

void parseInstructions(
    const std::string& function_name, // 函数名，用于匹配调试信息和字节码表中的函数名
    c10::ivalue::TupleElements&& ins_list, // 指令列表的元组元素，包含解析后的指令信息
    c10::ivalue::TupleElements& debug_handles_m_tuple, // 调试信息的元组元素，可能包含调试句柄和函数名
    mobile::Function* function) { // 指向移动函数对象的指针，用于构建函数指令

  c10::List<int64_t> debug_handles_list; // 存储调试句柄列表

  if (!debug_handles_m_tuple.empty()) {
    const std::string& debug_info_function_name =
        debug_handles_m_tuple[0].toStringRef(); // 调试信息中的函数名
    TORCH_CHECK(
        debug_info_function_name == function_name,
        "The function names in the bytecode table and the debug info table do not match."); // 检查字节码表和调试信息表中的函数名是否匹配

    IValue& debug_handles_table = debug_handles_m_tuple[1]; // 调试句柄表
    auto debugHandlesTableElements =
        std::move(*std::move(debug_handles_table).toTuple()).elements(); // 获取调试句柄表的元素

    // 从调试句柄表中提取函数调试句柄列表
    debug_handles_list = (expect_field(
                              debugHandlesTableElements,
                              "function_debug_handles",
                              BYTECODE_INDEX_MODULE_DEBUG_HANDLES)
                              .toTupleRef()
                              .elements())[0]
                             .toIntList();

    TORCH_CHECK(
        debug_handles_list.size() == ins_list.size(),
        "The numbers of instructions and debug handles strings do not match."); // 检查指令数和调试句柄数是否匹配
  }

  // NOTE: this won't perform particularly well if the ins_list IValue
  // didn't come from unpickler and thus have its strings
  // interned. Consider adding a flag to bypass the cache if that
  // becomes an important use case.

  OpCodeCache opCodeCache; // 操作码缓存对象，用于解析操作码

  // 遍历指令列表，解析并添加指令到函数对象
  for (const auto j : c10::irange(ins_list.size())) {
    auto ins_tuple = std::move(ins_list[j]).toTuple(); // 获取第 j 个指令的元组
    c10::ArrayRef<IValue> ins_item = ins_tuple->elements(); // 指令元组中的元素

    TORCH_CHECK(
        ins_item.size() == 3,
        "There should be three parts in an instruction. The function name is ",
        function_name); // 检查指令元组是否包含三个部分

    OpCode op_code = opCodeCache.parse(*ins_item[0].toString()); // 解析操作码
    int X = ins_item[1].toInt(); // X 参数
    int N = ins_item[2].toInt(); // N 参数

    // 如果存在调试句柄列表，则添加带调试句柄的指令，否则添加不带调试句柄的指令
    if (!debug_handles_list.empty()) {
      int64_t debug_handle = debug_handles_list[j];
      function->append_instruction(op_code, X, N, debug_handle);
    } else {
      function->append_instruction(op_code, X, N);
    }
  }
}

// 解析常量列表，将每个常量添加到函数对象中
void parseConstants(
    const c10::ivalue::TupleElements& consts_list,
    mobile::Function* function) {
  for (const auto& constant : consts_list) {
    function->append_constant(constant);
  }
}

// 解析类型列表，将每个类型添加到函数对象中
void parseTypes(
    const c10::ivalue::TupleElements& types_list,
    mobile::Function* function) {
  std::vector<std::string> types_string_list;
  types_string_list.resize(types_list.size());

  // 将类型元素转换为字符串列表
  for (size_t i = 0; i < types_list.size(); i++) {
    types_string_list[i] = types_list[i].toStringRef();
  }

  // 解析类型字符串列表为类型指针列表，并将每个类型指针添加到函数对象中
  std::vector<c10::TypePtr> types_ptr_list = c10::parseType(types_string_list);
  for (auto& type_ptr : types_ptr_list) {
    function->append_type(type_ptr);
  }
}

// 设置函数对象的寄存器大小
void parseRegisterSize(size_t rsize, mobile::Function* function) {
  function->set_register_size(rsize);
}
```