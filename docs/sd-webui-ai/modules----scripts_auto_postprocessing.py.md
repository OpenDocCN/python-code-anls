# `stable-diffusion-webui\modules\scripts_auto_postprocessing.py`

```py
# 从 modules 模块中导入 scripts、scripts_postprocessing、shared 三个模块
from modules import scripts, scripts_postprocessing, shared

# 定义 ScriptPostprocessingForMainUI 类，继承自 scripts.Script 类
class ScriptPostprocessingForMainUI(scripts.Script):
    # 初始化方法，接受 script_postproc 参数
    def __init__(self, script_postproc):
        # 将 script_postproc 赋值给 self.script
        self.script: scripts_postprocessing.ScriptPostprocessing = script_postproc
        # 初始化 postprocessing_controls 为 None
        self.postprocessing_controls = None

    # 返回脚本的标题
    def title(self):
        return self.script.name

    # 根据 is_img2img 返回是否显示脚本
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # 返回脚本的用户界面
    def ui(self, is_img2img):
        # 获取脚本的用户界面控件
        self.postprocessing_controls = self.script.ui()
        # 返回用户界面控件的值
        return self.postprocessing_controls.values()

    # 对图像进行后处理
    def postprocess_image(self, p, script_pp, *args):
        # 将控件和参数组成字典
        args_dict = dict(zip(self.postprocessing_controls, args))

        # 创建 PostprocessedImage 对象
        pp = scripts_postprocessing.PostprocessedImage(script_pp.image)
        pp.info = {}
        # 调用脚本的 process 方法进行处理
        self.script.process(pp, **args_dict)
        # 更新额外生成参数
        p.extra_generation_params.update(pp.info)
        # 更新脚本的图像
        script_pp.image = pp.image

# 创建自动预处理脚本数据
def create_auto_preprocessing_script_data():
    # 从 modules 模块中导入 scripts 模块
    from modules import scripts

    # 初始化结果列表
    res = []

    # 遍历需要在主界面启用的后处理脚本
    for name in shared.opts.postprocessing_enable_in_main_ui:
        # 查找对应脚本的数据
        script = next(iter([x for x in scripts.postprocessing_scripts_data if x.script_class.name == name]), None)
        # 如果脚本不存在，则继续下一次循环
        if script is None:
            continue

        # 使用 lambda 表达式创建 ScriptPostprocessingForMainUI 类的实例
        constructor = lambda s=script: ScriptPostprocessingForMainUI(s.script_class())
        # 将脚本数据添加到结果列表中
        res.append(scripts.ScriptClassData(script_class=constructor, path=script.path, basedir=script.basedir, module=script.module))

    # 返回结果列表
    return res
```