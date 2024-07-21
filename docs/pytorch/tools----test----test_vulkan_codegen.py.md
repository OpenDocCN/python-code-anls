# `.\pytorch\tools\test\test_vulkan_codegen.py`

```
# 导入必要的库和模块：tempfile用于创建临时文件和目录，unittest用于编写和运行单元测试
import tempfile
import unittest

# 导入自定义模块中的类和函数
from tools.gen_vulkan_spv import DEFAULT_ENV, SPVGenerator

####################
# Data for testing #
####################

# 测试用的着色器代码，包含了GLSL的代码和一些条件预处理指令
test_shader = """
#version 450 core

#define FORMAT ${FORMAT}
#define PRECISION ${PRECISION}
#define OP(X) ${OPERATOR}

$def is_int(dtype):
$   return dtype in {"int", "int32", "int8"}

$def is_uint(dtype):
$   return dtype in {"uint", "uint32", "uint8"}

$if is_int(DTYPE):
  #define VEC4_T ivec4
$elif is_uint(DTYPE):
  #define VEC4_T uvec4
$else:
  #define VEC4_T vec4

$if not INPLACE:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
$else:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict uimage3D uOutput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  $if not INPLACE:
    VEC4_T v = texelFetch(uInput, pos, 0);
  $else:
    VEC4_T v = imageLoad(uOutput, pos);
  $for i in range(ITER[0]):
    for (int i = 0; i < ${ITER[1]}; ++i) {
        v = OP(v + i);
    }
  imageStore(uOutput, pos, OP(v));
}

"""

# 测试用的参数配置文件，使用YAML格式描述了多个测试用例的参数和变体
test_params_yaml = """
test_shader:
  parameter_names_with_default_values:
    DTYPE: float
    INPLACE: false
    OPERATOR: X + 3
    ITER: !!python/tuple [3, 5]
  generate_variant_forall:
    INPLACE:
      - VALUE: false
        SUFFIX: ""
      - VALUE: true
        SUFFIX: inplace
    DTYPE:
      - VALUE: int8
      - VALUE: float
  shader_variants:
    - NAME: test_shader_1
    - NAME: test_shader_3
      OPERATOR: X - 1
      ITER: !!python/tuple [3, 2]
      generate_variant_forall:
        DTYPE:
        - VALUE: float
        - VALUE: int

"""

##############
# Unit Tests #
##############

# 单元测试类，继承自unittest.TestCase，用于测试Vulkan SPV代码生成器的功能
class TestVulkanSPVCodegen(unittest.TestCase):

    # 设置测试环境，在每个测试方法运行前调用
    def setUp(self) -> None:
        # 创建临时目录对象，用于存储测试用的GLSL代码和参数配置文件
        self.tmpdir = tempfile.TemporaryDirectory()

        # 将测试着色器代码写入临时文件
        with open(f"{self.tmpdir.name}/test_shader.glsl,", "w") as f:
            f.write(test_shader)

        # 将测试参数配置写入临时文件
        with open(f"{self.tmpdir.name}/test_params.yaml", "w") as f:
            f.write(test_params_yaml)

        # 创建另一个临时目录对象，用于存储生成的SPV文件
        self.tmpoutdir = tempfile.TemporaryDirectory()

        # 创建SPVGenerator对象，用于测试
        self.generator = SPVGenerator(
            src_dir_paths=self.tmpdir.name, env=DEFAULT_ENV, glslc_path=None
        )
    def cleanUp(self) -> None:
        # 清理临时目录和临时输出目录
        self.tmpdir.cleanup()
        self.tmpoutdir.cleanup()

    def testOutputMap(self) -> None:
        # 每个着色器变体将基于 DTYPE 和 INPLACE 参数的所有可能组合生成变体
        # test_shader_3 由于指定了自定义的 generate_variant_forall 字段，生成的变体较少
        expected_output_shaders = {
            "test_shader_1_float",
            "test_shader_1_inplace_float",
            "test_shader_1_inplace_int8",
            "test_shader_1_int8",
            "test_shader_3_float",
            "test_shader_3_int",
        }

        # 获取实际生成的着色器名称集合
        actual_output_shaders = set(self.generator.output_shader_map.keys())

        # 断言期望的输出着色器集合与实际生成的着色器集合相等
        self.assertEqual(expected_output_shaders, actual_output_shaders)
```