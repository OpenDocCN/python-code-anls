# `stable-diffusion-webui\extensions-builtin\hypertile\scripts\hypertile_xyz.py`

```py
# 从 modules 模块中导入 scripts 和 opts 模块
from modules import scripts
from modules.shared import opts

# 从 scripts_data 中筛选出 script_class 模块为 "xyz_grid.py" 的第一个模块，并将其赋值给 xyz_grid
xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

# 定义一个函数 int_applier，返回一个函数，用于将给定值应用到 opts.data 中的给定 value_name
def int_applier(value_name:str, min_range:int = -1, max_range:int = -1):
    """
    Returns a function that applies the given value to the given value_name in opts.data.
    """
    # 定义内部函数 validate，用于验证给定值是否符合要求
    def validate(value_name:str, value:str):
        # 将值转换为整数类型
        value = int(value)
        # 验证值是否在指定范围内
        if not min_range == -1:
            assert value >= min_range, f"Value {value} for {value_name} must be greater than or equal to {min_range}"
        if not max_range == -1:
            assert value <= max_range, f"Value {value} for {value_name} must be less than or equal to {max_range}"
    # 定义内部函数 apply_int，用于将整数值应用到 opts.data 中的 value_name
    def apply_int(p, x, xs):
        validate(value_name, x)
        opts.data[value_name] = int(x)
    return apply_int

# 定义一个函数 bool_applier，返回一个函数，用于将给定值应用到 opts.data 中的给定 value_name
def bool_applier(value_name:str):
    """
    Returns a function that applies the given value to the given value_name in opts.data.
    """
    # 定义内部函数 validate，用于验证给定值是否符合要求
    def validate(value_name:str, value:str):
        # 验证值是否为 "true" 或 "false"
        assert value.lower() in ["true", "false"], f"Value {value} for {value_name} must be either true or false"
    # 定义内部函数 apply_bool，用于将布尔值应用到 opts.data 中的 value_name
    def apply_bool(p, x, xs):
        validate(value_name, x)
        # 将值转换为布尔类型
        value_boolean = x.lower() == "true"
        opts.data[value_name] = value_boolean
    return apply_bool

# 定义一个函数 add_axis_options，用于添加轴选项
    # 定义额外的轴选项列表
    extra_axis_options = [
        # 创建一个轴选项对象，设置标签、数据类型、应用函数、选项列表
        xyz_grid.AxisOption("[Hypertile] Unet First pass Enabled", str, bool_applier("hypertile_enable_unet"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] Unet Second pass Enabled", str, bool_applier("hypertile_enable_unet_secondpass"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] Unet Max Depth", int, int_applier("hypertile_max_depth_unet", 0, 3), choices=lambda: [str(x) for x in range(4)]),
        xyz_grid.AxisOption("[Hypertile] Unet Max Tile Size", int, int_applier("hypertile_max_tile_unet", 0, 512)),
        xyz_grid.AxisOption("[Hypertile] Unet Swap Size", int, int_applier("hypertile_swap_size_unet", 0, 64)),
        xyz_grid.AxisOption("[Hypertile] VAE Enabled", str, bool_applier("hypertile_enable_vae"), choices=xyz_grid.boolean_choice(reverse=True)),
        xyz_grid.AxisOption("[Hypertile] VAE Max Depth", int, int_applier("hypertile_max_depth_vae", 0, 3), choices=lambda: [str(x) for x in range(4)]),
        xyz_grid.AxisOption("[Hypertile] VAE Max Tile Size", int, int_applier("hypertile_max_tile_vae", 0, 512)),
        xyz_grid.AxisOption("[Hypertile] VAE Swap Size", int, int_applier("hypertile_swap_size_vae", 0, 64)),
    ]
    # 获取当前轴选项集合和额外轴选项集合的标签集合
    set_a = {opt.label for opt in xyz_grid.axis_options}
    set_b = {opt.label for opt in extra_axis_options}
    # 如果当前轴选项集合和额外轴选项集合有交集，则直接返回，不做任何操作
    if set_a.intersection(set_b):
        return

    # 将额外轴选项列表添加到当前轴选项列表中
    xyz_grid.axis_options.extend(extra_axis_options)
```