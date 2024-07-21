# `.\pytorch\test\export\test_schema.py`

```
# 导入必要的模块和函数
from torch._export.serde.schema_check import (
    _Commit,            # 导入 _Commit 类
    _diff_schema,       # 导入 _diff_schema 函数
    check,              # 导入 check 函数
    SchemaUpdateError,  # 导入 SchemaUpdateError 异常类
    update_schema,      # 导入 update_schema 函数
)

from torch.testing._internal.common_utils import IS_FBCODE, run_tests, TestCase  # 导入必要的测试工具和类

# 定义测试类 TestSchema，继承自 TestCase 类
class TestSchema(TestCase):
    
    # 定义测试方法 test_schema_compatibility
    def test_schema_compatibility(self):
        # 提示信息，用于指导更新 schema
        msg = """
Detected an invalidated change to export schema. Please run the following script to update the schema:
Example(s):
    python scripts/export/update_schema.py --prefix <path_to_torch_development_diretory>
        """
        
        # 如果在 FBCODE 环境下，添加额外的更新 schema 的提示信息
        if IS_FBCODE:
            msg += """or
    buck run caffe2:export_update_schema -- --prefix /data/users/$USER/fbsource/fbcode/caffe2/
            """
        
        # 尝试执行更新 schema 的操作，并捕获 SchemaUpdateError 异常
        try:
            commit = update_schema()
        except SchemaUpdateError as e:
            # 如果更新失败，用异常信息和提示信息组成失败信息，并用 self.fail() 抛出失败
            self.fail(f"Failed to update schema: {e}\n{msg}")
        
        # 断言更新后的 checksum_base 和 checksum_result 相等，否则抛出失败，显示提示信息
        self.assertEqual(commit.checksum_base, commit.checksum_result, msg)
    
    # 定义测试方法 test_schema_diff
    def test_schema_diff(self):
        # 比较两个 schema 的差异，返回添加和删除的字段信息
        additions, subtractions = _diff_schema(
            {
                "Type0": {"kind": "struct", "fields": {}},  # 第一个 schema
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field0": {"type": ""},
                        "field2": {"type": ""},
                        "field3": {"type": "", "default": "[]"},
                    },
                },
            },
            {
                "Type2": {
                    "kind": "struct",
                    "fields": {
                        "field1": {"type": "", "default": "0"},
                        "field2": {"type": "", "default": "[]"},
                        "field3": {"type": ""},
                    },
                },
                "Type1": {"kind": "struct", "fields": {}},  # 第二个 schema
            },
        )
        
        # 断言添加的字段是否符合预期
        self.assertEqual(
            additions,
            {
                "Type1": {"kind": "struct", "fields": {}},  # 添加的 Type1 结构
                "Type2": {
                    "fields": {
                        "field1": {"type": "", "default": "0"},  # Type2 结构中添加的字段
                        "field2": {"default": "[]"},            # Type2 结构中修改的字段
                    },
                },
            },
        )
        
        # 断言删除的字段是否符合预期
        self.assertEqual(
            subtractions,
            {
                "Type0": {"kind": "struct", "fields": {}},  # 删除的 Type0 结构
                "Type2": {
                    "fields": {
                        "field0": {"type": ""},      # Type2 结构中删除的字段
                        "field3": {"default": "[]"},  # Type2 结构中修改的字段
                    },
                },
            },
        )

# 如果当前脚本是主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```