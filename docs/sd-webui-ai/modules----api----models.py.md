# `stable-diffusion-webui\modules\api\models.py`

```
# 导入 inspect 模块，用于获取对象信息
import inspect

# 导入必要的模块和类
from pydantic import BaseModel, Field, create_model
from typing import Any, Optional, Literal
from inflection import underscore
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import sd_upscalers, opts, parser

# 定义不允许的 API 参数列表
API_NOT_ALLOWED = [
    "self",
    "kwargs",
    "sd_model",
    "outpath_samples",
    "outpath_grids",
    "sampler_index",
    # "do_not_save_samples",
    # "do_not_save_grid",
    "extra_generation_params",
    "overlay_images",
    "do_not_reload_embeddings",
    "seed_enable_extras",
    "prompt_for_display",
    "sampler_noise_scheduler_override",
    "ddim_discretize"
]

# 定义辅助类 ModelDef，用于 Pydantic 动态模型生成
class ModelDef(BaseModel):
    """Assistance Class for Pydantic Dynamic Model Generation"""

    # 定义模型字段的属性
    field: str
    field_alias: str
    field_type: Any
    field_value: Any
    field_exclude: bool = False

# 定义 PydanticModelGenerator 类，用于处理类的生成
class PydanticModelGenerator:
    """
    Takes in created classes and stubs them out in a way FastAPI/Pydantic is happy about:
    source_data is a snapshot of the default values produced by the class
    params are the names of the actual keys required by __init__
    """

    # 初始化方法，接受模型名称、类实例和额外字段作为参数
    def __init__(
        self,
        model_name: str = None,
        class_instance = None,
        additional_fields = None,
    ):
        # 定义一个生成字段类型的函数，根据注解确定字段类型
        def field_type_generator(k, v):
            field_type = v.annotation

            # 如果字段类型为 'Image'，则将其转换为 'str'
            if field_type == 'Image':
                field_type = 'str'

            return Optional[field_type]

        # 合并类的参数，获取所有父类的参数
        def merge_class_params(class_):
            all_classes = list(filter(lambda x: x is not object, inspect.getmro(class_)))
            parameters = {}
            for classes in all_classes:
                parameters = {**parameters, **inspect.signature(classes.__init__).parameters}
            return parameters

        # 初始化模型名称和类参数
        self._model_name = model_name
        self._class_data = merge_class_params(class_instance)

        # 根据类参数生成模型定义列表
        self._model_def = [
            ModelDef(
                field=underscore(k),
                field_alias=k,
                field_type=field_type_generator(k, v),
                field_value=None if isinstance(v.default, property) else v.default
            )
            for (k,v) in self._class_data.items() if k not in API_NOT_ALLOWED
        ]

        # 添加额外字段到模型定义列表
        for fields in additional_fields:
            self._model_def.append(ModelDef(
                field=underscore(fields["key"]),
                field_alias=fields["key"],
                field_type=fields["type"],
                field_value=fields["default"],
                field_exclude=fields["exclude"] if "exclude" in fields else False))

    # 生成模型
    def generate_model(self):
        """
        Creates a pydantic BaseModel
        from the json and overrides provided at initialization
        """
        # 根据模型定义列表生成字段字典
        fields = {
            d.field: (d.field_type, Field(default=d.field_value, alias=d.field_alias, exclude=d.field_exclude)) for d in self._model_def
        }
        # 创建动态模型
        DynamicModel = create_model(self._model_name, **fields)
        # 允许通过字段名进行数据填充
        DynamicModel.__config__.allow_population_by_field_name = True
        # 允许对模型进行修改
        DynamicModel.__config__.allow_mutation = True
        return DynamicModel
# 生成用于处理文本到图像转换的 API 的 Pydantic 模型
StableDiffusionTxt2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingTxt2Img",
    StableDiffusionProcessingTxt2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
    ]
).generate_model()

# 生成用于处理图像到图像转换的 API 的 Pydantic 模型
StableDiffusionImg2ImgProcessingAPI = PydanticModelGenerator(
    "StableDiffusionProcessingImg2Img",
    StableDiffusionProcessingImg2Img,
    [
        {"key": "sampler_index", "type": str, "default": "Euler"},
        {"key": "init_images", "type": list, "default": None},
        {"key": "denoising_strength", "type": float, "default": 0.75},
        {"key": "mask", "type": str, "default": None},
        {"key": "include_init_images", "type": bool, "default": False, "exclude" : True},
        {"key": "script_name", "type": str, "default": None},
        {"key": "script_args", "type": list, "default": []},
        {"key": "send_images", "type": bool, "default": True},
        {"key": "save_images", "type": bool, "default": False},
        {"key": "alwayson_scripts", "type": dict, "default": {}},
    ]
).generate_model()

# 定义文本到图像转换的响应模型
class TextToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

# 定义图像到图像转换的响应模型
class ImageToImageResponse(BaseModel):
    images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    parameters: dict
    info: str

# 定义 ExtrasBaseRequest 类，作为基础请求模型
class ExtrasBaseRequest(BaseModel):
    # 设置 resize_mode 参数，用于指定调整大小的模式：0 表示按照 upscaling_resize 的倍数放大，1 表示放大到 upscaling_resize_h x upscaling_resize_w 的尺寸
    resize_mode: Literal[0, 1] = Field(default=0, title="Resize Mode", description="Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.")
    # 设置 show_extras_results 参数，用于指定是否返回生成的图像
    show_extras_results: bool = Field(default=True, title="Show results", description="Should the backend return the generated image?")
    # 设置 gfpgan_visibility 参数，用于指定 GFPGAN 的可见性，取值范围为 0 到 1
    gfpgan_visibility: float = Field(default=0, title="GFPGAN Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of GFPGAN, values should be between 0 and 1.")
    # 设置 codeformer_visibility 参数，用于指定 CodeFormer 的可见性，取值范围为 0 到 1
    codeformer_visibility: float = Field(default=0, title="CodeFormer Visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of CodeFormer, values should be between 0 and 1.")
    # 设置 codeformer_weight 参数，用于指定 CodeFormer 的权重，取值范围为 0 到 1
    codeformer_weight: float = Field(default=0, title="CodeFormer Weight", ge=0, le=1, allow_inf_nan=False, description="Sets the weight of CodeFormer, values should be between 0 and 1.")
    # 设置 upscaling_resize 参数，用于指定图像放大的倍数，仅在 resize_mode=0 时使用
    upscaling_resize: float = Field(default=2, title="Upscaling Factor", ge=1, le=8, description="By how much to upscale the image, only used when resize_mode=0.")
    # 设置 upscaling_resize_w 参数，用于指定目标宽度，仅在 resize_mode=1 时使用
    upscaling_resize_w: int = Field(default=512, title="Target Width", ge=1, description="Target width for the upscaler to hit. Only used when resize_mode=1.")
    # 设置 upscaling_resize_h 参数，用于指定目标高度，仅在 resize_mode=1 时使用
    upscaling_resize_h: int = Field(default=512, title="Target Height", ge=1, description="Target height for the upscaler to hit. Only used when resize_mode=1.")
    # 设置 upscaling_crop 参数，用于指定是否裁剪图像以适应所选尺寸
    upscaling_crop: bool = Field(default=True, title="Crop to fit", description="Should the upscaler crop the image to fit in the chosen size?")
    # 设置 upscaler_1 参数，用于指定主要的放大器名称，必须是 sd_upscalers 列表中的一个
    upscaler_1: str = Field(default="None", title="Main upscaler", description=f"The name of the main upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}")
    # 设置 upscaler_2 参数，用于指定次要的放大器名称，必须是 sd_upscalers 列表中的一个
    upscaler_2: str = Field(default="None", title="Secondary upscaler", description=f"The name of the secondary upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}")
    # 定义 extras_upscaler_2_visibility 变量，类型为 float，设置默认值为 0，标题为 "Secondary upscaler visibility"，取值范围为大于等于 0 小于等于 1，不允许无穷大或 NaN，描述为 "Sets the visibility of secondary upscaler, values should be between 0 and 1."
    extras_upscaler_2_visibility: float = Field(default=0, title="Secondary upscaler visibility", ge=0, le=1, allow_inf_nan=False, description="Sets the visibility of secondary upscaler, values should be between 0 and 1.")
    # 定义 upscale_first 变量，类型为 bool，设置默认值为 False，标题为 "Upscale first"，描述为 "Should the upscaler run before restoring faces?"
    upscale_first: bool = Field(default=False, title="Upscale first", description="Should the upscaler run before restoring faces?")
# 定义一个基础响应模型，包含一个字符串类型的 HTML 信息字段
class ExtraBaseResponse(BaseModel):
    html_info: str = Field(title="HTML info", description="A series of HTML tags containing the process info.")

# 定义一个单张图片请求模型，继承自 ExtrasBaseRequest，包含一个字符串类型的图片字段
class ExtrasSingleImageRequest(ExtrasBaseRequest):
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")

# 定义一个单张图片响应模型，继承自 ExtraBaseResponse，包含一个字符串类型的图片字段
class ExtrasSingleImageResponse(ExtraBaseResponse):
    image: str = Field(default=None, title="Image", description="The generated image in base64 format.")

# 定义一个文件数据模型，包含一个字符串类型的数据字段和一个字符串类型的文件名字段
class FileData(BaseModel):
    data: str = Field(title="File data", description="Base64 representation of the file")
    name: str = Field(title="File name")

# 定义一个批量图片请求模型，继承自 ExtrasBaseRequest，包含一个文件数据列表字段
class ExtrasBatchImagesRequest(ExtrasBaseRequest):
    imageList: list[FileData] = Field(title="Images", description="List of images to work on. Must be Base64 strings")

# 定义一个批量图片响应模型，继承自 ExtraBaseResponse，包含一个字符串类型的图片列表字段
class ExtrasBatchImagesResponse(ExtraBaseResponse):
    images: list[str] = Field(title="Images", description="The generated images in base64 format.")

# 定义一个 PNG 信息请求模型，包含一个字符串类型的图片字段
class PNGInfoRequest(BaseModel):
    image: str = Field(title="Image", description="The base64 encoded PNG image")

# 定义一个 PNG 信息响应模型，包含一个字符串类型的信息字段、一个字典类型的项目字段和一个字典类型的参数字段
class PNGInfoResponse(BaseModel):
    info: str = Field(title="Image info", description="A string with the parameters used to generate the image")
    items: dict = Field(title="Items", description="A dictionary containing all the other fields the image had")
    parameters: dict = Field(title="Parameters", description="A dictionary with parsed generation info fields")

# 定义一个进度请求模型，包含一个布尔类型的跳过当前图片字段
class ProgressRequest(BaseModel):
    skip_current_image: bool = Field(default=False, title="Skip current image", description="Skip current image serialization")

# 定义一个进度响应模型，包含一个浮点数类型的进度字段、一个浮点数类型的相对 ETA 字段和一个字典类型的状态字段
class ProgressResponse(BaseModel):
    progress: float = Field(title="Progress", description="The progress with a range of 0 to 1")
    eta_relative: float = Field(title="ETA in secs")
    state: dict = Field(title="State", description="The current state snapshot")
    # 当前图像的 base64 格式字符串，默认为 None
    current_image: str = Field(default=None, title="Current image", description="The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.")
    # 信息文本的字符串，默认为 None
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")
# 定义一个请求模型，包含图像和模型两个字段
class InterrogateRequest(BaseModel):
    # 图像字段，Base64 编码的图像数据
    image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")
    # 模型字段，用于指定使用的模型
    model: str = Field(default="clip", title="Model", description="The interrogate model used.")

# 定义一个响应模型，包含生成的图像描述字段
class InterrogateResponse(BaseModel):
    # 生成的图像描述
    caption: str = Field(default=None, title="Caption", description="The generated caption for the image.")

# 定义一个响应模型，用于训练任务的返回信息
class TrainResponse(BaseModel):
    # 训练信息
    info: str = Field(title="Train info", description="Response string from train embedding or hypernetwork task.")

# 定义一个响应模型，用于创建任务的返回信息
class CreateResponse(BaseModel):
    # 创建信息
    info: str = Field(title="Create info", description="Response string from create embedding or hypernetwork task.")

# 初始化一个空字典
fields = {}
# 遍历数据标签字典中的键值对
for key, metadata in opts.data_labels.items():
    # 获取数据字典中对应键的值
    value = opts.data.get(key)
    # 获取数据类型
    optType = opts.typemap.get(type(metadata.default), type(metadata.default)) if metadata.default else Any

    # 如果元数据不为空
    if metadata is not None:
        # 更新字段字典，包含字段名、类型和默认值等信息
        fields.update({key: (Optional[optType], Field(default=metadata.default, description=metadata.label))})
    else:
        # 更新字段字典，包含字段名和类型信息
        fields.update({key: (Optional[optType], Field())})

# 创建一个选项模型，包含各个字段的信息
OptionsModel = create_model("Options", **fields)

# 初始化一个空字典
flags = {}
# 获取参数解析器中的选项信息
_options = vars(parser)['_option_string_actions']
# 遍历选项信息
for key in _options:
    # 如果选项不是帮助选项
    if(_options[key].dest != 'help'):
        # 获取选项信息
        flag = _options[key]
        _type = str
        # 如果选项有默认值，则获取其类型
        if _options[key].default is not None:
            _type = type(_options[key].default)
        # 更新标志字典，包含标志名、类型和默认值等信息
        flags.update({flag.dest: (_type, Field(default=flag.default, description=flag.help))})

# 创建一个标志模型，包含各个标志的信息
FlagsModel = create_model("Flags", **flags)

# 定义一个采样器项模型，包含名称、别名和选项等字段
class SamplerItem(BaseModel):
    # 名称字段
    name: str = Field(title="Name")
    # 别名列表字段
    aliases: list[str] = Field(title="Aliases")
    # 选项字典字段
    options: dict[str, str] = Field(title="Options")

# 定义一个放大器项模型，包含名称、模型名称、模型路径和模型 URL 等字段
class UpscalerItem(BaseModel):
    # 名称字段
    name: str = Field(title="Name")
    # 模型名称字段
    model_name: Optional[str] = Field(title="Model Name")
    # 模型路径字段
    model_path: Optional[str] = Field(title="Path")
    # 模型 URL 字段
    model_url: Optional[str] = Field(title="URL")
    # 定义一个可选的浮点数类型的字段，字段名为"scale"，标题为"Scale"
    scale: Optional[float] = Field(title="Scale")
# 定义 LatentUpscalerModeItem 类，包含一个名为 name 的字符串字段
class LatentUpscalerModeItem(BaseModel):
    name: str = Field(title="Name")

# 定义 SDModelItem 类，包含 title、model_name、hash、sha256、filename 和 config 字段
class SDModelItem(BaseModel):
    title: str = Field(title="Title")
    model_name: str = Field(title="Model Name")
    hash: Optional[str] = Field(title="Short hash")
    sha256: Optional[str] = Field(title="sha256 hash")
    filename: str = Field(title="Filename")
    config: Optional[str] = Field(title="Config file")

# 定义 SDVaeItem 类，包含 model_name 和 filename 字段
class SDVaeItem(BaseModel):
    model_name: str = Field(title="Model Name")
    filename: str = Field(title="Filename")

# 定义 HypernetworkItem 类，包含 name 和 path 字段
class HypernetworkItem(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")

# 定义 FaceRestorerItem 类，包含 name 和 cmd_dir 字段
class FaceRestorerItem(BaseModel):
    name: str = Field(title="Name")
    cmd_dir: Optional[str] = Field(title="Path")

# 定义 RealesrganItem 类，包含 name、path 和 scale 字段
class RealesrganItem(BaseModel):
    name: str = Field(title="Name")
    path: Optional[str] = Field(title="Path")
    scale: Optional[int] = Field(title="Scale")

# 定义 PromptStyleItem 类，包含 name、prompt 和 negative_prompt 字段
class PromptStyleItem(BaseModel):
    name: str = Field(title="Name")
    prompt: Optional[str] = Field(title="Prompt")
    negative_prompt: Optional[str] = Field(title="Negative Prompt")

# 定义 EmbeddingItem 类，包含 step、sd_checkpoint、sd_checkpoint_name、shape 和 vectors 字段
class EmbeddingItem(BaseModel):
    step: Optional[int] = Field(title="Step", description="The number of steps that were used to train this embedding, if available")
    sd_checkpoint: Optional[str] = Field(title="SD Checkpoint", description="The hash of the checkpoint this embedding was trained on, if available")
    sd_checkpoint_name: Optional[str] = Field(title="SD Checkpoint Name", description="The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead")
    shape: int = Field(title="Shape", description="The length of each individual vector in the embedding")
    vectors: int = Field(title="Vectors", description="The number of vectors in the embedding")

# 定义 EmbeddingsResponse 类，为空
class EmbeddingsResponse(BaseModel):
    # 定义一个字典，用于存储当前模型加载的嵌入项
    loaded: dict[str, EmbeddingItem] = Field(title="Loaded", description="Embeddings loaded for the current model")
    # 定义一个字典，用于存储当前模型跳过的嵌入项（可能是由于架构不兼容导致）
    skipped: dict[str, EmbeddingItem] = Field(title="Skipped", description="Embeddings skipped for the current model (likely due to architecture incompatibility)")
# 定义内存响应的数据模型，包含 RAM 和 CUDA 两个字典类型字段
class MemoryResponse(BaseModel):
    ram: dict = Field(title="RAM", description="System memory stats")
    cuda: dict = Field(title="CUDA", description="nVidia CUDA memory stats")

# 定义脚本列表的数据模型，包含 txt2img 和 img2img 两个列表类型字段
class ScriptsList(BaseModel):
    txt2img: list = Field(default=None, title="Txt2img", description="Titles of scripts (txt2img)")
    img2img: list = Field(default=None, title="Img2img", description="Titles of scripts (img2img)")

# 定义脚本参数的数据模型，包含标签、值、最小值、最大值、步长和可选值等字段
class ScriptArg(BaseModel):
    label: str = Field(default=None, title="Label", description="Name of the argument in UI")
    value: Optional[Any] = Field(default=None, title="Value", description="Default value of the argument")
    minimum: Optional[Any] = Field(default=None, title="Minimum", description="Minimum allowed value for the argument in UI")
    maximum: Optional[Any] = Field(default=None, title="Minimum", description="Maximum allowed value for the argument in UI")
    step: Optional[Any] = Field(default=None, title="Minimum", description="Step for changing value of the argument in UI")
    choices: Optional[list[str]] = Field(default=None, title="Choices", description="Possible values for the argument")

# 定义脚本信息的数据模型，包含名称、是否总是运行、是否为 img2img 脚本、参数列表等字段
class ScriptInfo(BaseModel):
    name: str = Field(default=None, title="Name", description="Script name")
    is_alwayson: bool = Field(default=None, title="IsAlwayson", description="Flag specifying whether this script is an alwayson script")
    is_img2img: bool = Field(default=None, title="IsImg2img", description="Flag specifying whether this script is an img2img script")
    args: list[ScriptArg] = Field(title="Arguments", description="List of script's arguments")

# 定义扩展项的数据模型，包含名称、远程地址、分支和提交哈希等字段
class ExtensionItem(BaseModel):
    name: str = Field(title="Name", description="Extension name")
    remote: str = Field(title="Remote", description="Extension Repository URL")
    branch: str = Field(title="Branch", description="Extension Repository Branch")
    commit_hash: str = Field(title="Commit Hash", description="Extension Repository Commit Hash")
    # 定义一个字符串类型的属性 version，表示扩展的版本
    version: str = Field(title="Version", description="Extension Version")
    # 定义一个字符串类型的属性 commit_date，表示扩展的仓库提交日期
    commit_date: str = Field(title="Commit Date", description="Extension Repository Commit Date")
    # 定义一个布尔类型的属性 enabled，表示指定该扩展是否启用
    enabled: bool = Field(title="Enabled", description="Flag specifying whether this extension is enabled")
```