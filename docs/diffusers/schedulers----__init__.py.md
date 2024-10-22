# `.\diffusers\schedulers\__init__.py`

```py
# 版权声明，表明版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵循许可证，否则不得使用本文件。
# 可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非根据适用法律或书面协议另有规定，
# 否则根据许可证分发的软件是以“原样”方式提供的，
# 不附有任何明示或暗示的担保或条件。
# 请参阅许可证以了解管理权限和
# 限制的特定语言。

# 引入类型检查相关模块
from typing import TYPE_CHECKING

# 从 utils 模块引入多个工具函数和常量
from ..utils import (
    DIFFUSERS_SLOW_IMPORT,  # 用于慢导入的常量
    OptionalDependencyNotAvailable,  # 表示可选依赖不可用的异常
    _LazyModule,  # 延迟加载模块的工具
    get_objects_from_module,  # 从模块中获取对象的工具
    is_flax_available,  # 检查 Flax 库是否可用的函数
    is_scipy_available,  # 检查 SciPy 库是否可用的函数
    is_torch_available,  # 检查 PyTorch 库是否可用的函数
    is_torchsde_available,  # 检查 PyTorch SDE 是否可用的函数
)

# 初始化一个空字典用于存储虚拟模块
_dummy_modules = {}
# 初始化一个空字典用于存储导入结构
_import_structure = {}

# 尝试检查是否可用 PyTorch 库
try:
    if not is_torch_available():  # 如果 PyTorch 不可用
        raise OptionalDependencyNotAvailable()  # 抛出可选依赖不可用异常
except OptionalDependencyNotAvailable:  # 捕获可选依赖不可用异常
    from ..utils import dummy_pt_objects  # 引入虚拟 PyTorch 对象

    _dummy_modules.update(get_objects_from_module(dummy_pt_objects))  # 更新虚拟模块字典

else:  # 如果没有异常抛出
    # 定义导入结构的不同部分及其包含的调度器
    _import_structure["deprecated"] = ["KarrasVeScheduler", "ScoreSdeVpScheduler"]  # 过时的调度器
    _import_structure["scheduling_amused"] = ["AmusedScheduler"]  # Amused 调度器
    _import_structure["scheduling_consistency_decoder"] = ["ConsistencyDecoderScheduler"]  # 一致性解码器调度器
    _import_structure["scheduling_consistency_models"] = ["CMStochasticIterativeScheduler"]  # 一致性模型调度器
    _import_structure["scheduling_ddim"] = ["DDIMScheduler"]  # DDIM 调度器
    _import_structure["scheduling_ddim_cogvideox"] = ["CogVideoXDDIMScheduler"]  # CogVideoX DDIM 调度器
    _import_structure["scheduling_ddim_inverse"] = ["DDIMInverseScheduler"]  # DDIM 反向调度器
    _import_structure["scheduling_ddim_parallel"] = ["DDIMParallelScheduler"]  # DDIM 并行调度器
    _import_structure["scheduling_ddpm"] = ["DDPMScheduler"]  # DDPM 调度器
    _import_structure["scheduling_ddpm_parallel"] = ["DDPMParallelScheduler"]  # DDPM 并行调度器
    _import_structure["scheduling_ddpm_wuerstchen"] = ["DDPMWuerstchenScheduler"]  # DDPM Wuerstchen 调度器
    _import_structure["scheduling_deis_multistep"] = ["DEISMultistepScheduler"]  # DEIS 多步骤调度器
    _import_structure["scheduling_dpm_cogvideox"] = ["CogVideoXDPMScheduler"]  # CogVideoX DPM 调度器
    _import_structure["scheduling_dpmsolver_multistep"] = ["DPMSolverMultistepScheduler"]  # DPM 求解器多步骤调度器
    _import_structure["scheduling_dpmsolver_multistep_inverse"] = ["DPMSolverMultistepInverseScheduler"]  # DPM 求解器多步骤反向调度器
    _import_structure["scheduling_dpmsolver_singlestep"] = ["DPMSolverSinglestepScheduler"]  # DPM 求解器单步骤调度器
    _import_structure["scheduling_edm_dpmsolver_multistep"] = ["EDMDPMSolverMultistepScheduler"]  # EDM DPM 求解器多步骤调度器
    _import_structure["scheduling_edm_euler"] = ["EDMEulerScheduler"]  # EDM Euler 调度器
    _import_structure["scheduling_euler_ancestral_discrete"] = ["EulerAncestralDiscreteScheduler"]  # Euler 祖先离散调度器
    _import_structure["scheduling_euler_discrete"] = ["EulerDiscreteScheduler"]  # Euler 离散调度器
    _import_structure["scheduling_flow_match_euler_discrete"] = ["FlowMatchEulerDiscreteScheduler"]  # 流匹配 Euler 离散调度器
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_flow_match_heun_discrete"] = ["FlowMatchHeunDiscreteScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_heun_discrete"] = ["HeunDiscreteScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_ipndm"] = ["IPNDMScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_k_dpm_2_ancestral_discrete"] = ["KDPM2AncestralDiscreteScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_k_dpm_2_discrete"] = ["KDPM2DiscreteScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_lcm"] = ["LCMScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_pndm"] = ["PNDMScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_repaint"] = ["RePaintScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_sasolver"] = ["SASolverScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_sde_ve"] = ["ScoreSdeVeScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_tcd"] = ["TCDScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_unclip"] = ["UnCLIPScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_unipc_multistep"] = ["UniPCMultistepScheduler"]
    # 将调度器的名称映射到其对应的类，方便后续导入，包含多个调度器
    _import_structure["scheduling_utils"] = ["AysSchedules", "KarrasDiffusionSchedulers", "SchedulerMixin"]
    # 将调度器的名称映射到其对应的类，方便后续导入
    _import_structure["scheduling_vq_diffusion"] = ["VQDiffusionScheduler"]
try:
    # 检查 Flax 库是否可用
    if not is_flax_available():
        # 如果不可用，抛出自定义异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚假的 Flax 对象，避免依赖缺失的问题
    from ..utils import dummy_flax_objects  # noqa F403

    # 更新虚假模块，添加从虚假 Flax 对象中获取的内容
    _dummy_modules.update(get_objects_from_module(dummy_flax_objects))

else:
    # 更新导入结构，添加 Flax 相关的调度器
    _import_structure["scheduling_ddim_flax"] = ["FlaxDDIMScheduler"]
    _import_structure["scheduling_ddpm_flax"] = ["FlaxDDPMScheduler"]
    _import_structure["scheduling_dpmsolver_multistep_flax"] = ["FlaxDPMSolverMultistepScheduler"]
    _import_structure["scheduling_euler_discrete_flax"] = ["FlaxEulerDiscreteScheduler"]
    _import_structure["scheduling_karras_ve_flax"] = ["FlaxKarrasVeScheduler"]
    _import_structure["scheduling_lms_discrete_flax"] = ["FlaxLMSDiscreteScheduler"]
    _import_structure["scheduling_pndm_flax"] = ["FlaxPNDMScheduler"]
    _import_structure["scheduling_sde_ve_flax"] = ["FlaxScoreSdeVeScheduler"]
    _import_structure["scheduling_utils_flax"] = [
        "FlaxKarrasDiffusionSchedulers",
        "FlaxSchedulerMixin",
        "FlaxSchedulerOutput",
        "broadcast_to_shape_from_left",
    ]


try:
    # 检查 Torch 和 SciPy 库是否同时可用
    if not (is_torch_available() and is_scipy_available()):
        # 如果任何一个不可用，抛出自定义异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚假的 Torch 和 SciPy 对象，避免依赖缺失的问题
    from ..utils import dummy_torch_and_scipy_objects  # noqa F403

    # 更新虚假模块，添加从虚假对象中获取的内容
    _dummy_modules.update(get_objects_from_module(dummy_torch_and_scipy_objects))

else:
    # 更新导入结构，添加 LMS 调度器
    _import_structure["scheduling_lms_discrete"] = ["LMSDiscreteScheduler"]

try:
    # 检查 Torch 和 TorchSDE 库是否同时可用
    if not (is_torch_available() and is_torchsde_available()):
        # 如果任何一个不可用，抛出自定义异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚假的 Torch 和 TorchSDE 对象，避免依赖缺失的问题
    from ..utils import dummy_torch_and_torchsde_objects  # noqa F403

    # 更新虚假模块，添加从虚假对象中获取的内容
    _dummy_modules.update(get_objects_from_module(dummy_torch_and_torchsde_objects))

else:
    # 更新导入结构，添加 Cosine DPMSolver 多步调度器和 DPMSolver SDE 调度器
    _import_structure["scheduling_cosine_dpmsolver_multistep"] = ["CosineDPMSolverMultistepScheduler"]
    _import_structure["scheduling_dpmsolver_sde"] = ["DPMSolverSDEScheduler"]

# 检查类型注解或慢速导入标志
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    # 导入必要的工具函数和异常处理
    from ..utils import (
        OptionalDependencyNotAvailable,
        is_flax_available,
        is_scipy_available,
        is_torch_available,
        is_torchsde_available,
    )

    try:
        # 检查 Torch 库是否可用
        if not is_torch_available():
            # 如果不可用，抛出自定义异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚假的 PyTorch 对象，避免依赖缺失的问题
        from ..utils.dummy_pt_objects import *  # noqa F403
    else:  # 如果不满足之前的条件，则执行以下导入操作
        # 从 deprecated 模块导入 KarrasVeScheduler 和 ScoreSdeVpScheduler
        from .deprecated import KarrasVeScheduler, ScoreSdeVpScheduler
        # 从 scheduling_amused 模块导入 AmusedScheduler
        from .scheduling_amused import AmusedScheduler
        # 从 scheduling_consistency_decoder 模块导入 ConsistencyDecoderScheduler
        from .scheduling_consistency_decoder import ConsistencyDecoderScheduler
        # 从 scheduling_consistency_models 模块导入 CMStochasticIterativeScheduler
        from .scheduling_consistency_models import CMStochasticIterativeScheduler
        # 从 scheduling_ddim 模块导入 DDIMScheduler
        from .scheduling_ddim import DDIMScheduler
        # 从 scheduling_ddim_cogvideox 模块导入 CogVideoXDDIMScheduler
        from .scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
        # 从 scheduling_ddim_inverse 模块导入 DDIMInverseScheduler
        from .scheduling_ddim_inverse import DDIMInverseScheduler
        # 从 scheduling_ddim_parallel 模块导入 DDIMParallelScheduler
        from .scheduling_ddim_parallel import DDIMParallelScheduler
        # 从 scheduling_ddpm 模块导入 DDPMScheduler
        from .scheduling_ddpm import DDPMScheduler
        # 从 scheduling_ddpm_parallel 模块导入 DDPMParallelScheduler
        from .scheduling_ddpm_parallel import DDPMParallelScheduler
        # 从 scheduling_ddpm_wuerstchen 模块导入 DDPMWuerstchenScheduler
        from .scheduling_ddpm_wuerstchen import DDPMWuerstchenScheduler
        # 从 scheduling_deis_multistep 模块导入 DEISMultistepScheduler
        from .scheduling_deis_multistep import DEISMultistepScheduler
        # 从 scheduling_dpm_cogvideox 模块导入 CogVideoXDPMScheduler
        from .scheduling_dpm_cogvideox import CogVideoXDPMScheduler
        # 从 scheduling_dpmsolver_multistep 模块导入 DPMSolverMultistepScheduler
        from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        # 从 scheduling_dpmsolver_multistep_inverse 模块导入 DPMSolverMultistepInverseScheduler
        from .scheduling_dpmsolver_multistep_inverse import DPMSolverMultistepInverseScheduler
        # 从 scheduling_dpmsolver_singlestep 模块导入 DPMSolverSinglestepScheduler
        from .scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
        # 从 scheduling_edm_dpmsolver_multistep 模块导入 EDMDPMSolverMultistepScheduler
        from .scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
        # 从 scheduling_edm_euler 模块导入 EDMEulerScheduler
        from .scheduling_edm_euler import EDMEulerScheduler
        # 从 scheduling_euler_ancestral_discrete 模块导入 EulerAncestralDiscreteScheduler
        from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        # 从 scheduling_euler_discrete 模块导入 EulerDiscreteScheduler
        from .scheduling_euler_discrete import EulerDiscreteScheduler
        # 从 scheduling_flow_match_euler_discrete 模块导入 FlowMatchEulerDiscreteScheduler
        from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        # 从 scheduling_flow_match_heun_discrete 模块导入 FlowMatchHeunDiscreteScheduler
        from .scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
        # 从 scheduling_heun_discrete 模块导入 HeunDiscreteScheduler
        from .scheduling_heun_discrete import HeunDiscreteScheduler
        # 从 scheduling_ipndm 模块导入 IPNDMScheduler
        from .scheduling_ipndm import IPNDMScheduler
        # 从 scheduling_k_dpm_2_ancestral_discrete 模块导入 KDPM2AncestralDiscreteScheduler
        from .scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
        # 从 scheduling_k_dpm_2_discrete 模块导入 KDPM2DiscreteScheduler
        from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
        # 从 scheduling_lcm 模块导入 LCMScheduler
        from .scheduling_lcm import LCMScheduler
        # 从 scheduling_pndm 模块导入 PNDMScheduler
        from .scheduling_pndm import PNDMScheduler
        # 从 scheduling_repaint 模块导入 RePaintScheduler
        from .scheduling_repaint import RePaintScheduler
        # 从 scheduling_sasolver 模块导入 SASolverScheduler
        from .scheduling_sasolver import SASolverScheduler
        # 从 scheduling_sde_ve 模块导入 ScoreSdeVeScheduler
        from .scheduling_sde_ve import ScoreSdeVeScheduler
        # 从 scheduling_tcd 模块导入 TCDScheduler
        from .scheduling_tcd import TCDScheduler
        # 从 scheduling_unclip 模块导入 UnCLIPScheduler
        from .scheduling_unclip import UnCLIPScheduler
        # 从 scheduling_unipc_multistep 模块导入 UniPCMultistepScheduler
        from .scheduling_unipc_multistep import UniPCMultistepScheduler
        # 从 scheduling_utils 模块导入 AysSchedules, KarrasDiffusionSchedulers 和 SchedulerMixin
        from .scheduling_utils import AysSchedules, KarrasDiffusionSchedulers, SchedulerMixin
        # 从 scheduling_vq_diffusion 模块导入 VQDiffusionScheduler
        from .scheduling_vq_diffusion import VQDiffusionScheduler

    # 尝试检查 flax 是否可用
    try:
        # 如果 flax 不可用，则抛出异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 从 utils.dummy_flax_objects 模块导入所有内容，忽略 F403 警告
        from ..utils.dummy_flax_objects import *  # noqa F403
    else:  # 如果前面的条件不满足
        # 导入 FlaxDDIMScheduler 模块
        from .scheduling_ddim_flax import FlaxDDIMScheduler
        # 导入 FlaxDDPMScheduler 模块
        from .scheduling_ddpm_flax import FlaxDDPMScheduler
        # 导入 FlaxDPMSolverMultistepScheduler 模块
        from .scheduling_dpmsolver_multistep_flax import FlaxDPMSolverMultistepScheduler
        # 导入 FlaxEulerDiscreteScheduler 模块
        from .scheduling_euler_discrete_flax import FlaxEulerDiscreteScheduler
        # 导入 FlaxKarrasVeScheduler 模块
        from .scheduling_karras_ve_flax import FlaxKarrasVeScheduler
        # 导入 FlaxLMSDiscreteScheduler 模块
        from .scheduling_lms_discrete_flax import FlaxLMSDiscreteScheduler
        # 导入 FlaxPNDMScheduler 模块
        from .scheduling_pndm_flax import FlaxPNDMScheduler
        # 导入 FlaxScoreSdeVeScheduler 模块
        from .scheduling_sde_ve_flax import FlaxScoreSdeVeScheduler
        # 导入调度相关的工具函数和类
        from .scheduling_utils_flax import (
            FlaxKarrasDiffusionSchedulers,  # Karras 扩散调度器
            FlaxSchedulerMixin,  # 调度器混合类
            FlaxSchedulerOutput,  # 调度器输出类
            broadcast_to_shape_from_left,  # 从左侧广播形状
        )

    try:  # 尝试检查必要的依赖项
        # 如果 PyTorch 和 SciPy 不可用，则抛出异常
        if not (is_torch_available() and is_scipy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:  # 捕获依赖项不可用的异常
        # 从 dummy_torch_and_scipy_objects 模块导入所有内容，避免引入实际依赖
        from ..utils.dummy_torch_and_scipy_objects import *  # noqa F403
    else:  # 如果没有异常，则导入 LMSDiscreteScheduler 模块
        from .scheduling_lms_discrete import LMSDiscreteScheduler

    try:  # 再次尝试检查其他必要依赖项
        # 如果 PyTorch 和 torchsde 不可用，则抛出异常
        if not (is_torch_available() and is_torchsde_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:  # 捕获依赖项不可用的异常
        # 从 dummy_torch_and_torchsde_objects 模块导入所有内容，避免引入实际依赖
        from ..utils.dummy_torch_and_torchsde_objects import *  # noqa F403
    else:  # 如果没有异常，则导入 CosineDPMSolverMultistepScheduler 和 DPMSolverSDEScheduler 模块
        from .scheduling_cosine_dpmsolver_multistep import CosineDPMSolverMultistepScheduler
        from .scheduling_dpmsolver_sde import DPMSolverSDEScheduler
# 否则执行以下代码
else:
    # 导入 sys 模块以访问系统特性
    import sys

    # 将当前模块替换为延迟加载的模块实例，使用模块名、文件名、导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 遍历虚拟模块字典，将每个模块名和对应的值设置到当前模块
    for name, value in _dummy_modules.items():
        setattr(sys.modules[__name__], name, value)
```