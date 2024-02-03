# `stable-diffusion-webui\modules\textual_inversion\ui.py`

```
# 导入 html 模块
import html
# 导入 gradio 模块并重命名为 gr
import gradio as gr
# 导入 modules.textual_inversion.textual_inversion 模块
import modules.textual_inversion.textual_inversion
# 从 modules 中导入 sd_hijack 和 shared 模块
from modules import sd_hijack, shared

# 创建嵌入，接受名称、初始化文本、nvpt、是否覆盖旧数据作为参数
def create_embedding(name, initialization_text, nvpt, overwrite_old):
    # 调用 textual_inversion 模块的 create_embedding 函数创建嵌入
    filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)

    # 调用 model_hijack 模块的 embedding_db.load_textual_inversion_embeddings 函数
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    # 返回下拉菜单的更新和创建信息
    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""

# 训练嵌入，接受任意数量的参数
def train_embedding(*args):
    # 断言不使用低内存选项
    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    # 获取是否应用优化的标志
    apply_optimizations = shared.opts.training_xattention_optimizations
    try:
        # 如果不应用优化，则撤销优化
        if not apply_optimizations:
            sd_hijack.undo_optimizations()

        # 调用 textual_inversion 模块的 train_embedding 函数训练嵌入
        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)

        # 构建结果字符串
        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        # 返回结果字符串和空字符串
        return res, ""
    except Exception:
        # 抛出异常
        raise
    finally:
        # 如果不应用优化，则应用优化
        if not apply_optimizations:
            sd_hijack.apply_optimizations()
```