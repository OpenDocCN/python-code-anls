# `.\pytorch\torch\ao\quantization\fx\_model_report\detector.py`

```
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Any, Dict, Set, Tuple, Callable, List

# 引入 PyTorch 相关库
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat

# 引入抽象基类
from abc import ABC, abstractmethod

# 引入量化相关模块和类
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
    QConfig,
    default_qconfig,
    _assert_valid_qconfig,
)
from torch.ao.quantization.observer import (
    ObserverBase,
    default_dynamic_quant_observer,
    default_per_channel_weight_observer,
    default_observer,
    default_weight_observer,
)
from torch.ao.quantization.fx._equalize import (
    default_equalization_qconfig,
    EqualizationQConfig,
)
from torch.ao.quantization.observer import _is_activation_post_process

# 定义检测器插入的关键字
DETECTOR_TARGET_NODE_KEY = "target_node"       # 目标节点键名
DETECTOR_OBS_TO_INSERT_KEY = "observer_to_insert"  # 要插入的观察器键名
DETECTOR_IS_POST_OBS_KEY = "is_post_observer"  # 是否为后观察器键名
DETECTOR_OBS_ARGS_KEY = "observer_args"         # 观察器参数键名

# 映射相关的代码
class DetectorQConfigInfo:
    r"""
    This class contains the QConfig information for a single module.
    The list of variables / values this contains can grow depending on the
    extensibility of the qconfig mapping feature set but this currently includes:
    - if activation observer is dynamic
    - if weight observer is per channel

    Args:
        module_fqn (str): The fully qualified name (fqn) of the module that this
            information contains info relevant to qconfig for
    """

    def __init__(self, module_fqn: str):
        super().__init__()
        self.module_fqn = module_fqn

        # 填充此部分以包含所有可能重要的变量
        # 如果您的检测器实际上在使用这些信息，请将其从“none”更改为适当的值
        self.is_activation_dynamic = False   # 激活观察器是否动态
        self.is_weight_per_channel = False   # 权重观察器是否按通道
        self.is_equalization_recommended = False   # 均衡建议相关选项
    def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
        r"""
        Args:
            module (torch.nn.Module) The module we are generating
            the qconfig for

        Returns the generated quantization QConfig according to what a valid configuration is
        """
        # 应用建议到新的量化配置
        module_qconfig = default_qconfig

        # 保持动态和按通道推荐的跟踪
        recommendations_list = []
        # 将组合作为列表附加
        recommendations_list.append((self.is_activation_dynamic, self.is_weight_per_channel))
        recommendations_list.append((self.is_activation_dynamic, False))  # 只尝试动态推荐
        recommendations_list.append((False, self.is_weight_per_channel))  # 只尝试按通道推荐

        # 现在我们尝试每个组合
        for rec in recommendations_list:
            # rec[0] -> 推荐动态
            # rec[1] -> 推荐按通道
            activation = default_dynamic_quant_observer if rec[0] else default_observer
            weight = default_per_channel_weight_observer if rec[1] else default_weight_observer
            test_config = QConfig(activation, weight)
            try:
                _assert_valid_qconfig(test_config, module)
                module_qconfig = test_config
                break
            except AssertionError:
                # 如果不是有效的配置，则继续尝试下一个优先级的配置
                continue

        # 返回所选的 QConfig
        return module_qconfig

    def generate_equalization_qconfig(self) -> EqualizationQConfig:
        r"""
        This returns the equalization configuration for a module.

        For now, it just returns the default, but as more equalization options become
        possible, this method can get more fleshed out with more nuanced granularity.


        Returns the generated equalization QConfig according to what a valid configuration is
        """
        # 在这种情况下，我们只返回默认的均衡配置
        # 我们知道这是有效的，因为只有有效的模块才会有这个选项
        return default_equalization_qconfig
# 添加检测器的基类
class DetectorBase(ABC):
    r""" Base Detector Module
    Any detector class should derive from this class.

    Concrete detectors should follow the same general API, which includes:
    - A method to calculate and return observer insertion points
        - Should return both the fqns and the Observer class to insert
    - A method to return a report based on the detector
        - Should return a str-based report and dict info in Tuple[str,Dict] format
    """

    def __init__(self):
        super().__init__()
        self.detector_config_info = None  # 初始化检测器配置信息为None

    @abstractmethod
    def determine_observer_insert_points(self, model) -> Dict:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict.
            This dict maps string keys to detector specific information
        """
        pass  # 抽象方法，具体的检测器需要实现该方法来确定插入观察器的位置和类别

    @abstractmethod
    def get_detector_name(self) -> str:
        r""" Returns the name of the current detector """
        pass  # 抽象方法，返回当前检测器的名称

    @abstractmethod
    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        r""" Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        pass  # 抽象方法，返回每个模块FQN相关的DetectorQConfigInfo，用于生成特定模块的QConfig
    # 定义一个方法来获取目标节点
    def _get_targeting_node(self, prepared_fx_model: GraphModule, target_fqn: str) -> torch.fx.node.Node:
        r"""
        Takes in a GraphModule and the target_fqn and finds the node whose target is this fqn.

        If it's not found, it means it is most likely inside a fused layer
            We just go one layer up in terms of the fqn we are searching for until we find parent node
            If we get to empty string, then we know that it doesn't exist

        The reason for the recursion is that if the model that we are looking for got fused,
        we will have module fqn as e.g. x.linear.0 but the graph will only have a node for the fused module,
        which would have fqn as x.linear so they will not match.
        To handle this, if we don't match, we then take off the last bit of the fqn e.g. x.linear.0 -> x.linear,
        or more generally foo.bar.baz -> foo.bar and search again, this will allow us to locate the correct module
        even in cases with fusion

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule
            target_fqn (str): The fqn of the layer we are trying to target

        Returns the node object we are trying to add observers around
        """
        
        # 遍历图中的所有节点
        for node in prepared_fx_model.graph.nodes:
            # 如果节点的目标与目标fqn相匹配，则返回该节点
            if node.target == target_fqn:
                return node

        # 如果走到这一步说明未找到目标节点
        # 如果目标fqn中没有"."，则已经到达基本层级且失败
        parent_fqn_sep_index = target_fqn.rfind(".")
        if parent_fqn_sep_index == -1:
            # 抛出值错误，表示目标fqn在图的目标中未找到
            raise ValueError("passed in target_fqn not found in graph's targets.")
        else:
            # 递归调用自身，使用父级fqn再次尝试查找
            return self._get_targeting_node(prepared_fx_model, target_fqn[:parent_fqn_sep_index])

    @abstractmethod
    def generate_detector_report(self, model) -> Tuple[str, Dict[str, Any]]:
        r"""
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Tuple of two elements:
            Str: string report of the suggested improvements
            Dict: contains useful data collected by the observer pertinent to this report
        """
        pass
# 定义一个名为 PerChannelDetector 的类，继承自 DetectorBase 类
class PerChannelDetector(DetectorBase):
    r""" This class is used to detect if any Linear or Conv layers in a model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        per_channel quantization can lead to major benefits in the form of accuracy.
        Therefore, if the backend used by the user supports it, it is recommended to use

        Args:
            backend (str, optional): the backend the user wishes to use in production
                Default value is current torch.backends.quantized.engine
    """

    # 用于返回字典的键
    BACKEND_KEY = "backend"
    PER_CHAN_SUPPORTED_KEY = "per_channel_quantization_supported"
    PER_CHAN_USED_KEY = "per_channel_quantization_used"

    # 默认映射，表示不同后端支持的每通道量化模块集合
    DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES: Dict[str, Set[Any]] = {
        "fbgemm": {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d},
        "qnnpack": {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d},
        "onednn": {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d},
        "x86": {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nnqat.Linear, nnqat.Conv1d, nnqat.Conv2d, nnqat.Conv3d},
    }

    # 初始化方法，接受一个后端参数，默认为 torch.backends.quantized.engine
    def __init__(self, backend: str = torch.backends.quantized.engine):
        # 调用父类的初始化方法
        super().__init__()

        # 存储后端信息
        self.backend_chosen = backend
        # 支持的模块集合，默认为空集合
        self.supported_modules = set()
        # 如果选择的后端在默认映射中有定义，则将支持的模块设置为该后端支持的模块集合
        if self.backend_chosen in self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES:
            self.supported_modules = self.DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES[self.backend_chosen]
        else:
            # 如果选择的后端不在默认映射中，则抛出 ValueError 异常
            raise ValueError(f"Not configured to work with {self.backend_chosen}. Try a different default backend")

    # 返回检测器的名称字符串
    def get_detector_name(self) -> str:
        r""" returns the string name of this detector"""
        return "per_channel_detector"
    # 返回一个字典，将每个模块的完全限定名映射到 DetectorQConfigInfo 对象
    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        r""" Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points
    
        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # 运行辅助函数来填充字典
        per_channel_info = self._detect_per_channel_helper(model)
    
        # 实际上我们正在填充一个 qconfig info 对象
        module_fqn_to_detector_qconfig_info = {}
    
        # 遍历每个模块的完全限定名
        for module_fqn in per_channel_info:
            # 创建一个 DetectorQConfigInfo 实例
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)
    
            # 检查是否支持按通道量化
            per_chan_supported: bool = per_channel_info[module_fqn][self.PER_CHAN_SUPPORTED_KEY]
            detector_qconfig_info.is_weight_per_channel = per_chan_supported
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info
    
        # 返回模块完全限定名到 DetectorQConfigInfo 对象的映射
        return module_fqn_to_detector_qconfig_info
    
    
    # 确定观察器插入点的方法
    def determine_observer_insert_points(self, model: nn.Module) -> Dict:
        r"""
        There is no observers inserted for the PerChannelDetector.
    
        Returns an empty dictionary since no observers are added or needed
        """
        # 返回一个空字典，因为不需要添加或使用观察器
        return {}
    def _detect_per_channel_helper(self, model: nn.Module):
        r"""
        determines if per_channel quantization is supported in modules and submodules.

        Returns a dictionary in the higher level _detect_per_channel function.
        Each entry maps the fully-qualified-name to information on whether per_channel quantization.

        Args:
            model: The current module that is being checked to see if it is per_channel quantizable

        Returns dictionary mapping fqns to if per_channel quantization is possible
        """
        # create dict we will return
        per_channel_info: Dict = {}

        # get the fully qualified name and check if in list of modules to include and list of modules to ignore
        for fqn, module in model.named_modules():

            is_in_include_list = any(isinstance(module, x) for x in self.supported_modules)

            # check if the module per_channel is supported
            # based on backend
            per_channel_supported = False

            if is_in_include_list:
                per_channel_supported = True

                # assert statement for MyPy
                q_config_file = module.qconfig
                assert isinstance(q_config_file, QConfig)

                # this object should either be fake quant or observer
                q_or_s_obj = module.qconfig.weight.p.func()
                assert isinstance(q_or_s_obj, (FakeQuantize, ObserverBase))

                per_channel_used = False  # will be true if found in qconfig

                if hasattr(q_or_s_obj, "ch_axis"):  # then we know that per_channel quantization used

                    # all fake quants have channel axis so need to check is_per_channel
                    if isinstance(q_or_s_obj, FakeQuantize):
                        if hasattr(q_or_s_obj, "is_per_channel") and q_or_s_obj.is_per_channel:
                            per_channel_used = True
                    elif isinstance(q_or_s_obj, ObserverBase):
                        # should be an observer otherwise
                        per_channel_used = True
                    else:
                        raise ValueError("Should be either observer or fake quant")

                # store information in per_channel_info dictionary
                per_channel_info[fqn] = {
                    self.PER_CHAN_SUPPORTED_KEY: per_channel_supported,
                    self.PER_CHAN_USED_KEY: per_channel_used,
                    self.BACKEND_KEY: self.backend_chosen
                }

        return per_channel_info
    def generate_detector_report(self, model: nn.Module) -> Tuple[str, Dict[str, Any]]:
        r"""Checks if any Linear or Conv layers in the model utilize per_channel quantization.
        Only Linear and Conv layers can use per_channel as of now so only these two are currently checked.

        Looks at q_config format and backend to determine if per_channel can be utilized.
        Uses the DEFAULT_BACKEND_PER_CHANNEL_SUPPORTED_MODULES structure to determine support

        Args:
            model: The prepared and calibrated model we want to check if using per_channel

        Returns a tuple with two elements:
            String report of potential actions to improve model (if per_channel quantization is available in backend)
            Dictionary mapping per_channel quantizable elements to:
                whether per_channel quantization is supported by the backend
                if it is being utilized in the current model
        """

        # 运行辅助函数以填充字典
        per_channel_info = self._detect_per_channel_helper(model)

        # 字符串，用于提示用户进行进一步的优化
        further_optims_str = f"Further Optimizations for backend {self.backend_chosen}: \n"

        # 是否存在可能的优化
        optimizations_possible = False
        for fqn in per_channel_info:
            fqn_dict = per_channel_info[fqn]
            # 检查是否后端支持 per_channel，并且当前模型未使用 per_channel
            if fqn_dict[self.PER_CHAN_SUPPORTED_KEY] and not fqn_dict[self.PER_CHAN_USED_KEY]:
                optimizations_possible = True
                further_optims_str += f"Module {fqn} can be configured to use per_channel quantization.\n"

        # 如果存在可能的优化，则说明如何配置 qconfig 以支持 per_channel 量化
        if optimizations_possible:
            further_optims_str += (
                "To use per_channel quantization, make sure the qconfig has a per_channel weight observer."
            )
        else:
            further_optims_str += "No further per_channel optimizations possible."

        # 返回包含字符串和相同信息字典形式的元组
        return (further_optims_str, per_channel_info)
# 定义一个名为 DynamicStaticDetector 的类，继承自 DetectorBase 类
class DynamicStaticDetector(DetectorBase):
    """
    确定给定模块是否更适合使用动态量化还是静态量化。

    利用 ModelReportObserver 记录范围信息。
    数据的稳态分布严格高于比较统计量的容差级别：

        S = average_batch_activation_range / epoch_activation_range

    非稳态分布在此度量标准的容差水平以下或者在其上。

    如果模块后的数据分布是非稳态的，则建议使用动态量化。
    否则建议使用静态量化。

    Args:
        tolerance (float, optional): S 指标稳态以上和非稳态以下的阈值。默认值: 0.5
    """

    # 插入的前后观察器的默认名称
    DEFAULT_PRE_OBSERVER_NAME = "model_report_pre_observer"
    DEFAULT_POST_OBSERVER_NAME = "model_report_post_observer"

    # 稳态和非稳态数据的命名约定
    STATIONARY_STR = "stationary"
    NON_STATIONARY_STR = "non-stationary"

    # 输入激活和输出激活的命名前缀
    INPUT_ACTIVATION_PREFIX = "input_activation_"
    OUTPUT_ACTIVATION_PREFIX = "output_activation_"

    # 返回模块信息键的命名约定
    TOLERANCE_KEY = "dynamic_static_tolerance"
    DEFAULT_DYNAMIC_REC_KEY = "dynamic_recommended"
    PRE_OBS_COMP_STAT_KEY = INPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    POST_OBS_COMP_STAT_KEY = OUTPUT_ACTIVATION_PREFIX + "dynamic_static_comp_stat"
    PRE_OBS_DATA_DIST_KEY = INPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    POST_OBS_DATA_DIST_KEY = OUTPUT_ACTIVATION_PREFIX + "dynamic_static_data_classification"
    IS_CURRENTLY_SUPPORTED_KEY = "is_dynamic_supported"

    # 对于此报告函数同时支持动态和静态的模块
    DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED = {nn.Linear}

    # 将来会支持动态和静态的模块
    DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED = {nn.Conv1d, nn.Conv2d, nn.Conv3d}

    # 类的初始化方法
    def __init__(self, tolerance=0.5):
        super().__init__()

        # 设置容差级别，并初始化一个集合以跟踪有用的观察器全限定名位置
        self.tolerance = tolerance
        self.useful_observer_fqns: Set[str] = set()
    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        r"""
        Determines where observers need to be inserted for the Dynamic vs Static detector.
        For this detector, we want to place observers on either side of linear layers in the model.

        Currently inserts observers for:
            linear layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver  # 使用 ModelReportObserver 类作为观察器对象

        # return dict
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}  # 创建空字典，用于存储观察器的全限定名及相关信息

        for fqn, module in prepared_fx_model.named_modules():  # 遍历模型中的所有模块及其全限定名
            # make sure module is supported
            if self._is_supported(module, insert=True):  # 检查当前模块是否支持插入观察器
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)  # 获取目标节点

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME  # 创建前置观察器的全限定名

                obs_fqn_to_info[pre_obs_fqn] = {  # 将前置观察器的信息加入字典
                    DETECTOR_TARGET_NODE_KEY: targeted_node,  # 目标节点
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),  # 要插入的观察器对象
                    DETECTOR_IS_POST_OBS_KEY: False,  # 表明这是前置观察器
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args  # 观察器的参数
                }

                # add entry for post-observer
                post_obs_fqn = fqn + "." + self.DEFAULT_POST_OBSERVER_NAME  # 创建后置观察器的全限定名

                obs_fqn_to_info[post_obs_fqn] = {  # 将后置观察器的信息加入字典
                    DETECTOR_TARGET_NODE_KEY: targeted_node,  # 目标节点
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(),  # 要插入的观察器对象
                    DETECTOR_IS_POST_OBS_KEY: True,  # 表明这是后置观察器
                    DETECTOR_OBS_ARGS_KEY: (targeted_node,)  # 观察器的参数
                }

        return obs_fqn_to_info  # 返回包含观察器信息的字典

    def get_detector_name(self) -> str:
        r""" returns the string name of this detector"""
        return "dynamic_vs_static_detector"  # 返回当前检测器的名称
    # 返回每个模块的 DetectorQConfigInfo，用于量化配置
    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        r""" Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # 运行辅助函数，填充字典
        dynamic_static_info = self._generate_dict_info(model)

        # 实际上我们正在填充一个 qconfig 信息对象
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in dynamic_static_info:
            # 创建一个 DetectorQConfigInfo 实例
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # 检查是否支持通道量化
            dynamic_static_recommended: bool = dynamic_static_info[module_fqn][self.DEFAULT_DYNAMIC_REC_KEY]
            detector_qconfig_info.is_activation_dynamic = dynamic_static_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info

    # 检查给定模块是否支持观察器
    def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # 检查模块是否属于支持的类型之一
        is_supported_type = any(isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_CHECK_SUPPORTED)

        # 检查模块是否将来可能支持
        future_supported_type = any(isinstance(module, x) for x in self.DEFAULT_DYNAMIC_STATIC_FUTURE_SUPPORTED)

        # 判断是否支持
        supported = is_supported_type or future_supported_type

        # 如果是用于插入观察器的检查
        if insert:
            return supported
        else:
            # 如果是用于生成报告，还需要检查模块是否包含观察器
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME) and hasattr(module, self.DEFAULT_POST_OBSERVER_NAME)
            return supported and has_obs
class InputWeightEqualizationDetector(DetectorBase):
    r"""
    Determines whether input-weight equalization can help improve quantization for certain modules.

    Specifically, this list of modules includes:
        linear
        conv

    Determines whether input-weight equalization is recommended based on the comp stat:
        s_c = sqrt(w_c/W)/sqrt(i_c/I)
        where:
            w_c is range of weight for channel c, W is range of weight over all channels
            i_c is range of input for channel c, I is range of input over all channels

        if s_c >= threshold or <= 1 / threshold, recommends input-weight equalization

    Args:
        ratio_threshold (float): The threshold for s_c to determine if input-weight equalization is suggested
            Should be between 0 and 1 (both non-inclusive)
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for s_c to determine if input-weight equalization is suggested
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine input weight equalization

    * :attr:`SUPPORTED_MODULES`: This specifies the modules that are supported for input-weight equalization

    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: The name of the pre-observer to be inserted for this detector
    """

    SUPPORTED_MODULES: Set[Callable] = {nn.Linear,
                                        nn.Conv1d,
                                        nn.Conv2d,
                                        nn.Conv3d,
                                        nnqat.Linear,
                                        nnqat.Conv1d,
                                        nnqat.Conv2d,
                                        nnqat.Conv3d}

    # names for the pre and post observers that are inserted
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"

    # weight / activation prefix for each of the below info
    WEIGHT_PREFIX = "weight_"
    ACTIVATION_PREFIX = "input_activation_"

    # string names for keys of info dictionaries
    PER_CHANNEL_MAX_KEY = "per_channel_max"
    PER_CHANNEL_MIN_KEY = "per_channel_min"
    GLOBAL_MAX_KEY = "global_max"
    GLOBAL_MIN_KEY = "global_min"

    # keys for return dict of recommendations
    RECOMMENDED_KEY = "input_weight_equalization_recommended"
    COMP_METRIC_KEY = "input_weight_channel_comparison_metrics"
    THRESHOLD_KEY = "input_weight_threshold"
    CHANNEL_KEY = "input_weight_channel_axis"

    # default weight and info strings
    WEIGHT_STR = "weight"
    INPUT_STR = "input"

    # default for what ratio we recommend input weight
    DEFAULT_RECOMMEND_INPUT_WEIGHT_CHANNEL_RATIO = 0.4
    def __init__(self, ratio_threshold: float, ch_axis: int = 1):
        # ensure passed in inputs are valid
        if ratio_threshold <= 0 or ratio_threshold >= 1:
            raise ValueError("Make sure threshold is > 0 and < 1")

        # initialize attributes based on args
        self.ratio_threshold: float = ratio_threshold
        self.ch_axis: int = ch_axis



        def _is_supported(self, module: nn.Module, insert: bool = False) -> bool:
        r"""Returns whether the given module is supported for observers

        Args
            module: The module to check and ensure is supported
            insert: True if this is check for observer insertion, false if for report gen

        Returns True if the module is supported by observer, False otherwise
        """
        # check to see if module is of a supported type
        is_supported_type = any(type(module) is x for x in self.SUPPORTED_MODULES)

        # this is check for observer insertion
        if insert:
            return is_supported_type
        else:
            # this is for report gen and we also need to check if it contains observers
            has_obs = hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)
            return is_supported_type and has_obs



        def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        r""" Returns the DetectorQConfigInfo for each module_fqn relevant
        Args
            model (nn.Module or subclass): model to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A DetectorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # run the helper function to populate the dictionary
        # find the range of inputs
        input_values: Dict[str, Dict] = self._extract_input_info(model)

        # find the range of weights
        weight_values: Dict[str, Dict] = self._extract_weight_info(model)

        # calculate per_channel comparison statistic s_c
        comp_stats: Dict[str, torch.Tensor] = self._generate_comparison_values(input_values, weight_values)

        # generate the return dictionary
        input_weight_equalization_info: Dict[str, Dict] = self._generate_dict_info(input_values, weight_values, comp_stats)

        # we actually have a qconfig info object we are populating
        module_fqn_to_detector_qconfig_info = {}

        for module_fqn in input_weight_equalization_info:
            # create a detector info instance
            detector_qconfig_info = DetectorQConfigInfo(module_fqn)

            # see if per channel quantization is supported
            input_weight_recommended: bool = input_weight_equalization_info[module_fqn][self.RECOMMENDED_KEY]
            detector_qconfig_info.is_equalization_recommended = input_weight_recommended
            module_fqn_to_detector_qconfig_info[module_fqn] = detector_qconfig_info

        return module_fqn_to_detector_qconfig_info
    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        r"""Determines where observers need to be inserted for the Input Weight Equalization Detector.
        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            linear layers
            conv layers

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """

        # observer for this detector is ModelReportObserver
        obs_ctr = ModelReportObserver

        # return dict
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}

        for fqn, module in prepared_fx_model.named_modules():
            # check to see if module is of a supported type
            if self._is_supported(module, insert=True):
                # if it's a supported type, we want to get node and add observer insert locations
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # add entry for pre-observer
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,  # Node to observe
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis),  # Observer instance to insert
                    DETECTOR_IS_POST_OBS_KEY: False,  # Indicates this is a pre-observer
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,  # Arguments for the observer
                }

        return obs_fqn_to_info

    def get_detector_name(self) -> str:
        r"""Returns the name of this detector"""
        return "input_weight_equalization_detector"
    # 定义一个方法用于从校准后的 GraphModule 提取输入信息
    def _extract_input_info(self, model: GraphModule) -> Dict[str, Dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the input information for each observer returns it

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping relevant module fqns (str) to a dict with keys:
            "input_activation_per_channel_max" : maps to the per_channel max values
            "input_activation_per_channel_min" : maps to the per_channel min values
            "input_activation_global_max" : maps to the global max recorded
            "input_activation_global_min" : maps to the global min recorded
        """

        # 初始化一个空字典，用于存储输入信息
        input_info: Dict[str, Dict] = {}

        # 遍历模型中的所有模块及其完全限定名
        for fqn, module in model.named_modules():
            # 如果模块是被支持的，并且它有一个预观察器
            if self._is_supported(module):
                # 获取模块的预观察器
                pre_obs = getattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

                # 将模块的输入信息存储到字典中
                input_info[fqn] = {
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MAX_KEY: pre_obs.max_val,
                    self.ACTIVATION_PREFIX + self.PER_CHANNEL_MIN_KEY: pre_obs.min_val,
                    self.ACTIVATION_PREFIX + self.GLOBAL_MAX_KEY: max(pre_obs.max_val),
                    self.ACTIVATION_PREFIX + self.GLOBAL_MIN_KEY: min(pre_obs.min_val),
                }

        # 返回包含输入信息的字典
        return input_info
    def _extract_weight_info(self, model: GraphModule) -> Dict[str, Dict]:
        r"""
        Takes in a calibrated GraphModule and then finds the relevant observers.
        It then extracts the weight information for each layer an observer is attached to.

        Args
            model (GraphModule): The prepared and calibrated GraphModule with inserted ModelReportObservers

        Returns a dict mapping module fqns (str) to a dict with keys:
            "per_channel_max" : maps to the per_channel max values
            "per_channel_min" : maps to the per_channel min values
            "global_max" : maps to the global max recorded
            "global_min" : maps to the global min recorded
        """
        # return dictionary mapping observer fqns to desired info
        weight_info: Dict[str, Dict] = {}

        # Iterate over all modules in the model
        for fqn, module in model.named_modules():
            # Check if the module is supported and has a pre-observer attached
            if self._is_supported(module):
                # Extract weights from the module
                # Calculate minimum and maximum values of the weights
                device = module.weight.device
                min_val: torch.Tensor = torch.tensor([float('inf')], device=device)
                max_val: torch.Tensor = torch.tensor([float('-inf')], device=device)
                x_copy = module.weight
                x_dim = x_copy.size()

                # Create a new permutation of axes
                new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
                new_axis_list[self.ch_axis] = 0
                new_axis_list[0] = self.ch_axis
                y = x_copy.permute(new_axis_list)

                # Ensure dtype consistency for min_val and max_val
                y = y.to(min_val.dtype)
                y = torch.flatten(y, start_dim=1)

                # Update min_val and max_val based on flattened tensor y
                if min_val.numel() == 0 or max_val.numel() == 0:
                    min_val, max_val = torch.aminmax(y, dim=1)
                else:
                    min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                    min_val = torch.min(min_val_cur, min_val)
                    max_val = torch.max(max_val_cur, max_val)

                # Store the extracted weight information in weight_info dictionary
                weight_info[fqn] = {
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MAX_KEY: max_val,
                    self.WEIGHT_PREFIX + self.PER_CHANNEL_MIN_KEY: min_val,
                    self.WEIGHT_PREFIX + self.GLOBAL_MAX_KEY: max(max_val),
                    self.WEIGHT_PREFIX + self.GLOBAL_MIN_KEY: min(min_val),
                }

        # Return the final weight_info dictionary
        return weight_info
    def _calculate_range_ratio(self, info_dict: Dict, info_str: str, module_fqn: str) -> torch.Tensor:
        r"""
        Takes in an info dict and calculates the s_c matrix.

        Args:
            info_dict (dict): A dictionary of either input or weight range info
            info_str (str): A str describing whether currently looking at weight or input info
                Either "weight" or "input"
            module_fqn (str): The fqn of the module we are looking at

        Returns a tensor of values, where each value is the s_c stat for a different channel
        """
        # calculate the ratios of the info

        # Determine the prefix string based on whether we are processing input or weight info
        prefix_str = self.ACTIVATION_PREFIX if info_str == self.INPUT_STR else self.WEIGHT_PREFIX

        # Calculate the range per channel by subtracting min from max values
        per_channel_range = info_dict[prefix_str + self.PER_CHANNEL_MAX_KEY] - info_dict[prefix_str + self.PER_CHANNEL_MIN_KEY]
        
        # Calculate the global range by subtracting global min from global max values
        global_range = info_dict[prefix_str + self.GLOBAL_MAX_KEY] - info_dict[prefix_str + self.GLOBAL_MIN_KEY]

        # Check if the global range is zero, indicating a constant value channel
        if global_range == 0:
            range_zero_explanation = "We recommend removing this channel as it doesn't provide any useful information."
            # Raise a ValueError with an explanation message
            raise ValueError(
                f"The range of the {info_str} data for module {module_fqn} is 0, "
                f"which means you have a constant value channel. {range_zero_explanation}"
            )

        # Calculate the ratio of per-channel range to global range
        ratio = per_channel_range / global_range

        # Return the calculated ratio as a tensor
        return ratio
    def _generate_comparison_values(self, input_info: Dict, weight_info: Dict) -> Dict[str, torch.Tensor]:
        r"""
        Takes in the information on the min and max values of the inputs and weights and:
            Calculates the comp stat for each channel: s_c = sqrt(w_c/W)/sqrt(i_c/I)

        Args:
            input_info (dict): A dict mapping each observer to input range information
            weight_info (dict): A dict mapping each observer to weight range information

        Returns a dict mapping relevant observer fqns (str) to a 1-D tensor.
            Each value is a different s_c value for a different channel
        """
        # create return dictionary for each observer
        module_fqn_to_channel: Dict[str, torch.Tensor] = {}

        # for each module (both passed in dicts should have same keys)
        for module_fqn in input_info:

            # raise error if not in weight info
            if module_fqn not in weight_info:
                raise KeyError(f"Unable to find weight range stats for module {module_fqn}")

            # calculate the ratios of the weight info and input info
            weight_ratio = self._calculate_range_ratio(weight_info[module_fqn], self.WEIGHT_STR, module_fqn)
            input_ratio = self._calculate_range_ratio(input_info[module_fqn], self.INPUT_STR, module_fqn)

            # if mismatched size, because of grouping, we want to replicate weight enough times
            weight_channels = len(weight_ratio)
            input_channels = len(input_ratio)
            if weight_channels != input_channels:
                # we try to replicate
                assert input_channels % weight_channels == 0, "input channels should be divisible by weight channels."
                # get replication factor
                rep_factor: int = input_channels // weight_channels

                # weight ratio is (n,), input ratio is (k,), we just repeat weight ratio k // n
                weight_ratio = weight_ratio.repeat(rep_factor)

            # calculate the s metric per channel
            s = torch.sqrt(weight_ratio) / torch.sqrt(input_ratio)
            module_fqn_to_channel[module_fqn] = s

        # return compiled observer ratios
        return module_fqn_to_channel
    def _generate_dict_info(self, input_info: Dict, weight_info: Dict, comp_stats: Dict) -> Dict[str, Dict]:
        r"""
        Helper function for generate_detector_report that does the generation of the dictionary.
        This process is done as specified in generate_detector_report documentation

        Args:
            input_info (dict): A dict mapping each module to input range information
            weight_info (dict): A dict mapping each module to weight range information
            comp_stats (dict): A dict mapping each module to its corresponding comp stat

        Returns a dictionary mapping each module with relevant ModelReportObservers around them to:
            whether input weight equalization is recommended
            their s_c metric compared to the threshold
            the threshold used to make the recommendation
            the channel used for recording data
            the input channel range info
            the weight channel range info
        """
        # 存储模块的输入权重均衡信息
        input_weight_equalization_info: Dict[str, Dict] = {}

        # 对每个模块添加单独的建议集
        for module_fqn in input_info:

            # 获取此模块的相关信息
            mod_input_info: Dict = input_info[module_fqn]
            mod_weight_info: Dict = weight_info[module_fqn]
            mod_comp_stat: Dict = comp_stats[module_fqn]

            # 判断每个通道是否应该进行输入权重均衡
            channel_rec_vals: list = []

            for val in mod_comp_stat:
                float_rep: float = val.item()

                # 判断是否建议进行输入权重均衡
                recommended: bool = float_rep >= self.ratio_threshold and float_rep <= 1 / self.ratio_threshold
                channel_rec_vals.append(recommended)

            # 构建返回字典输入，同时将输入和权重字典解包到其中
            input_weight_equalization_info[module_fqn] = {
                self.RECOMMENDED_KEY: channel_rec_vals,
                self.COMP_METRIC_KEY: mod_comp_stat,
                self.THRESHOLD_KEY: self.ratio_threshold,
                self.CHANNEL_KEY: self.ch_axis,
                **mod_input_info,
                **mod_weight_info,
            }

        # 返回每个模块的编译信息
        return input_weight_equalization_info
class OutlierDetector(DetectorBase):
    r"""
    Determines whether there are significant outliers in activation data around a certain layer.

    This is ideally used in conjunction with information on stationary vs. non-stationary distribution:
        If the data is stationary, and there are significant outliers, then we want to flag them
        We want to do this on a per channel basis for detecting outliers

    Determines whether activation data is flagged as outlier based on if data is stationary and:
        p_r = avg(100th percentile / "reference_percentile"th percentile)
        where:
            p_r is average percentile ratio across all batches in the epoch
            reference_percentile is a percentile values between 0 and 100 exclusive

        if p_r is above some threshold, then we consider the activations to have significant outliers

    Args:
        ratio_threshold (float, optional): The threshold for p_r to determine if there are outliers in activations
            Should be >= 1
            Default: 3.5
        reference_percentile (float, optional): The denominator to find the relative scale of the 100th percentile
            Should be between 0 and 1
            Default: 0.975
        fraction_batches_used_threshold (float, optional): Threshold of fraction of batches per channel to determine outlier
            If fraction is below this, we deem number of samples used to calculate outliers as insignificant and alert user
            regardless of whether we detected outliers or not in channel to take a closer look at channel results
            Should be between 0 and 1
            Default: 0.95
        ch_axis (int, optional): The channel axis being observed to determine input weight equalization
            Default: 1

    * :attr:`ratio_threshold`: The threshold for p_r to determine if there are outliers in activations
        The p_r value (average ratio of 100th percentile/reference_percentile) is compared to ratio_threshold
        If it is significantly greater, then we consider it an outlier
        This threshold was calculated based on the ratio of the percentiles in a normal distribution
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`reference_percentile`: The denominator of the top fraction to find the relative scale of the 100th percentile
        Should be between 0 and 1
        The calculations behind value choice: https://drive.google.com/file/d/1N2wdtXWI-kOH8S7HH4-PYB_NmqzZil4p/view?usp=sharing

    * :attr:`fraction_batches_used_threshold`: The fraction of batches to determine outliers for each channel should be above this
        Some batches may not be used because of 0-based errors, so this is to ensure a good amount of the total batches are used
        Should be between 0 and 1

    * :attr:`ch_axis`: The channel axis being observed to determine outliers
        Specifies the axis along which channels are observed for outlier detection, default is 1
    """
    * :attr:`DEFAULT_PRE_OBSERVER_NAME`: 要为此检测器插入的预观察器的名称
    """

    # 用于插入的预观察器的名称
    DEFAULT_PRE_OBSERVER_NAME: str = "model_report_pre_observer"

    # 输入激活前缀
    INPUT_ACTIVATION_PREFIX = "input_activation_"

    # 字典键的名称
    OUTLIER_KEY = "outliers_detected"
    NUM_BATCHES_KEY = "outlier_detection_batches_used"
    IS_SUFFICIENT_BATCHES_KEY = "outlier_detection_is_sufficient_batches"
    COMP_METRIC_KEY = "outlier_detection_percentile_ratios"
    RATIO_THRES_KEY = "outlier_detection_ratio_threshold"
    REF_PERCENTILE_KEY = "outlier_detection_reference_percentile"
    CHANNEL_AXIS_KEY = "outlier_detection_channel_axis"
    MAX_VALS_KEY = INPUT_ACTIVATION_PREFIX + "per_channel_max"
    CONSTANT_COUNTS_KEY = "constant_batch_counts"

    def __init__(
        self,
        ratio_threshold: float = 3.5,
        reference_percentile: float = 0.975,
        fraction_batches_used_threshold: float = 0.95,
        ch_axis: int = 1,
    ):
        # 初始化感兴趣的变量
        self.ratio_threshold = ratio_threshold

        # 确保传入的百分位数是有效的
        assert reference_percentile >= 0 and reference_percentile <= 1
        assert fraction_batches_used_threshold >= 0 and fraction_batches_used_threshold <= 1
        self.reference_percentile = reference_percentile
        self.fraction_batches_used_threshold = fraction_batches_used_threshold
        self.ch_axis = ch_axis

    def get_detector_name(self) -> str:
        r"""返回此检测器的名称"""
        return "outlier_detector"

    def _supports_insertion(self, module: nn.Module) -> bool:
        r"""返回给定模块是否支持观察器插入

        任何没有子模块且不是观察器本身的模块都是支持的

        Args
            module: 要检查并确保支持的模块

        Returns 如果模块支持观察器，则返回True，否则返回False
        """
        # 插入模块的情况
        # 检查模块是否有子模块且不是后处理激活的观察器
        num_children = len(list(module.children()))
        return num_children == 0 and not _is_activation_post_process(module)

    def get_qconfig_info(self, model) -> Dict[str, DetectorQConfigInfo]:
        r"""返回每个模块_fqn相关的DetectorQConfigInfo

        Args
            model (nn.Module或其子类): 要找到插入观察器点的模型

        Returns 一个字典，将唯一的观察器fqn（我们想要在哪里插入它们）映射到：
            一个DetectorQConfigInfo，其中包含为特定模块生成QConfig所需的信息
        """
        # 目前对异常检测器不执行任何操作
        return {}
    # 判断给定模块是否支持报告生成
    def _supports_report_gen(self, module: nn.Module) -> bool:
        r"""Returns whether the given module is supported for report generation

        Any module that has a model report pre-observer is supported

        Args
            module: The module to check and ensure is supported

        Returns True if the module is supported by observer, False otherwise
        """
        # 检查模块是否具有指定名称的属性，用于判断是否支持报告生成
        return hasattr(module, self.DEFAULT_PRE_OBSERVER_NAME)

    # 确定插入观察器的位置
    def determine_observer_insert_points(self, prepared_fx_model: GraphModule) -> Dict[str, Dict[str, Any]]:
        r""" Determines where observers need to be inserted for the Outlier Detector.

        For this detector, we want to place observers in front of supported layers.

        Currently inserts observers for:
            all layers that do not have children (leaf level layers)

        Args:
            prepared_fx_model (GraphModule):  The prepared Fx GraphModule

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to a Dict with:
            key "target_node" -> the node we are trying to observe with this observer (torch.fx.node.Node)
            key "observer_to_insert" -> the observer we wish to insert (ObserverBase)
            key "is_post_observer" -> True if this is meant to be a post-observer for target_node, False if pre-observer
            key "observer_args" -> The arguments that are meant to be passed into the observer
        """
        # 使用 ModelReportObserver 作为观察器
        obs_ctr = ModelReportObserver

        # 用于存储观察器信息的字典
        obs_fqn_to_info: Dict[str, Dict[str, Any]] = {}

        # 遍历预处理后的图模块中的所有模块
        for fqn, module in prepared_fx_model.named_modules():
            # 检查模块是否支持插入观察器
            if self._supports_insertion(module):
                # 获取目标节点以及添加观察器的插入位置
                targeted_node = self._get_targeting_node(prepared_fx_model, fqn)

                # 创建前置观察器的完全限定名称
                pre_obs_fqn = fqn + "." + self.DEFAULT_PRE_OBSERVER_NAME

                # 将前置观察器信息添加到字典中
                obs_fqn_to_info[pre_obs_fqn] = {
                    DETECTOR_TARGET_NODE_KEY: targeted_node,
                    DETECTOR_OBS_TO_INSERT_KEY: obs_ctr(ch_axis=self.ch_axis, comp_percentile=self.reference_percentile),
                    DETECTOR_IS_POST_OBS_KEY: False,
                    DETECTOR_OBS_ARGS_KEY: targeted_node.args,
                }

        # 返回观察器信息的字典
        return obs_fqn_to_info

    # 计算异常信息
    def _calculate_outlier_info(
        self,
        percentile_ratios: torch.Tensor,
        counted_batches: torch.Tensor,
        total_batches: int,
    ) -> Dict[str, List[bool]]:
        r"""
        给出百分位比率是否被视为异常值的信息
        同时提供数据是否具有统计显著性来支持此断言的信息

        Args:
            percentile_ratios (torch.Tensor): 观察者计算的每个通道的平均百分位比率
            counted_batches (torch.Tensor): 用于平均计算的张量每批次的数量
            total_batches (int): 在本轮中观察者通过的总批次数

        返回一个字典，映射如下：
            "outliers_detected" : 每个通道的布尔值列表，如果被视为异常值则为True
            "is_sufficient_batches": 如果 o_r >= fraction_batches_used_threshold，则为True：
                其中 o_r = counted_batches / total_batches
        """
        outlier_dict: Dict[str, List[bool]] = {self.OUTLIER_KEY: [], self.IS_SUFFICIENT_BATCHES_KEY: []}

        # 将百分位比率和批次数转换为扁平化列表，便于映射
        ratios_list: List = percentile_ratios.tolist()
        num_batches_list: List = counted_batches.tolist()

        # 计算每个通道是否具有统计显著性
        significant_size = [
            batch_size / total_batches >= self.fraction_batches_used_threshold for batch_size in num_batches_list
        ]
        outlier_dict[self.IS_SUFFICIENT_BATCHES_KEY] = significant_size

        # 根据比率计算每个通道是否为异常值
        outlier_detected = [ratio > self.ratio_threshold for ratio in ratios_list]
        outlier_dict[self.OUTLIER_KEY] = outlier_detected

        # 返回包含两个列表的字典
        return outlier_dict
```