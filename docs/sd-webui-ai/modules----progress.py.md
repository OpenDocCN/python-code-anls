# `stable-diffusion-webui\modules\progress.py`

```
# 导入必要的库
import base64
import io
import time

# 导入 gradio 库中的模块
import gradio as gr
# 导入 pydantic 库中的 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入 shared 模块中的 opts 变量
from modules.shared import opts

# 导入 shared 模块并重命名为 shared
import modules.shared as shared

# 当前任务的 ID
current_task = None
# 待处理任务的字典，键为任务 ID，值为任务添加时间
pending_tasks = {}
# 已完成任务的列表
finished_tasks = []
# 记录的结果列表
recorded_results = []
# 记录结果的限制
recorded_results_limit = 2

# 开始处理任务
def start_task(id_task):
    global current_task

    current_task = id_task
    # 从待处理任务中移除当前任务
    pending_tasks.pop(id_task, None)

# 完成任务
def finish_task(id_task):
    global current_task

    # 如果当前任务是指定的任务 ID
    if current_task == id_task:
        current_task = None

    # 将任务 ID 添加到已完成任务列表中
    finished_tasks.append(id_task)
    # 如果已完成任务列表长度超过 16，则移除最早的任务
    if len(finished_tasks) > 16:
        finished_tasks.pop(0)

# 记录任务结果
def record_results(id_task, res):
    # 将任务 ID 和结果添加到记录结果列表中
    recorded_results.append((id_task, res))
    # 如果记录结果列表长度超过限制，则移除最早的记录
    if len(recorded_results) > recorded_results_limit:
        recorded_results.pop(0)

# 将任务添加到待处理队列中
def add_task_to_queue(id_job):
    # 将任务 ID 和当前时间添加到待处理任务字典中
    pending_tasks[id_job] = time.time()

# 进度请求的数据模型
class ProgressRequest(BaseModel):
    id_task: str = Field(default=None, title="Task ID", description="id of the task to get progress for")
    id_live_preview: int = Field(default=-1, title="Live preview image ID", description="id of last received last preview image")
    live_preview: bool = Field(default=True, title="Include live preview", description="boolean flag indicating whether to include the live preview image")

# 进度响应的数据模型
class ProgressResponse(BaseModel):
    active: bool = Field(title="Whether the task is being worked on right now")
    queued: bool = Field(title="Whether the task is in queue")
    completed: bool = Field(title="Whether the task has already finished")
    progress: float = Field(default=None, title="Progress", description="The progress with a range of 0 to 1")
    eta: float = Field(default=None, title="ETA in secs")
    live_preview: str = Field(default=None, title="Live preview image", description="Current live preview; a data: uri")
    id_live_preview: int = Field(default=None, title="Live preview image ID", description="Send this together with next request to prevent receiving same image")
    # 定义一个字符串类型的属性textinfo，用于存储信息文本，用于WebUI
    textinfo: str = Field(default=None, title="Info text", description="Info text used by WebUI.")
# 设置进度 API 的路由，指定处理函数为 progressapi，请求方法为 POST，响应模型为 ProgressResponse
def setup_progress_api(app):
    return app.add_api_route("/internal/progress", progressapi, methods=["POST"], response_model=ProgressResponse)

# 进度 API 的处理函数，根据请求中的任务 ID 判断任务状态，并返回相应的进度信息
def progressapi(req: ProgressRequest):
    # 判断当前任务是否为活动状态
    active = req.id_task == current_task
    # 判断当前任务是否在待处理任务列表中
    queued = req.id_task in pending_tasks
    # 判断当前任务是否在已完成任务列表中
    completed = req.id_task in finished_tasks

    # 如果任务不是活动状态
    if not active:
        # 设置默认文本信息为 "Waiting..."
        textinfo = "Waiting..."
        # 如果任务在待处理任务列表中
        if queued:
            # 对待处理任务列表按照任务优先级排序
            sorted_queued = sorted(pending_tasks.keys(), key=lambda x: pending_tasks[x])
            # 获取当前任务在排序后的队列中的位置
            queue_index = sorted_queued.index(req.id_task)
            # 更新文本信息为 "In queue: 当前任务位置/总任务数"
            textinfo = "In queue: {}/{}".format(queue_index + 1, len(sorted_queued))
        # 返回进度信息对象
        return ProgressResponse(active=active, queued=queued, completed=completed, id_live_preview=-1, textinfo=textinfo)

    # 初始化进度值为 0
    progress = 0

    # 获取共享状态中的任务总数和当前任务编号
    job_count, job_no = shared.state.job_count, shared.state.job_no
    # 获取共享状态中的采样步数和当前采样步
    sampling_steps, sampling_step = shared.state.sampling_steps, shared.state.sampling_step

    # 如果任务总数大于 0
    if job_count > 0:
        # 根据当前任务编号和任务总数计算进度
        progress += job_no / job_count
    # 如果采样步数大于 0 且任务总数大于 0
    if sampling_steps > 0 and job_count > 0:
        # 根据任务总数、采样步和采样步数计算进度
        progress += 1 / job_count * sampling_step / sampling_steps

    # 将进度限制在 0 到 1 之间
    progress = min(progress, 1)

    # 计算任务开始以来经过的时间
    elapsed_since_start = time.time() - shared.state.time_start
    # 预测任务完成所需的时间
    predicted_duration = elapsed_since_start / progress if progress > 0 else None
    # 计算预计剩余时间
    eta = predicted_duration - elapsed_since_start if predicted_duration is not None else None

    # 初始化实时预览为 None
    live_preview = None
    # 获取实时预览的任务 ID
    id_live_preview = req.id_live_preview
    # 如果启用了实时预览并且请求中包含实时预览
    if opts.live_previews_enable and req.live_preview:
        # 设置当前图像状态
        shared.state.set_current_image()
        # 如果当前实时预览的 ID 不同于请求中的实时预览 ID
        if shared.state.id_live_preview != req.id_live_preview:
            # 获取当前图像
            image = shared.state.current_image
            # 如果图像不为空
            if image is not None:
                # 创建一个字节流缓冲区
                buffered = io.BytesIO()

                # 根据图像格式设置保存参数
                if opts.live_previews_image_format == "png":
                    # 对于较大的图像，使用 optimize 会花费大量时间
                    if max(*image.size) <= 256:
                        save_kwargs = {"optimize": True}
                    else:
                        save_kwargs = {"optimize": False, "compress_level": 1}

                else:
                    save_kwargs = {}

                # 将图像保存到缓冲区中
                image.save(buffered, format=opts.live_previews_image_format, **save_kwargs)
                # 将缓冲区中的数据转换为 base64 编码的字符串
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                # 构建实时预览的数据 URI
                live_preview = f"data:image/{opts.live_previews_image_format};base64,{base64_image}"
                # 获取实时预览的 ID
                id_live_preview = shared.state.id_live_preview

    # 返回包含进度信息、实时预览等内容的响应对象
    return ProgressResponse(active=active, queued=queued, completed=completed, progress=progress, eta=eta, live_preview=live_preview, id_live_preview=id_live_preview, textinfo=shared.state.textinfo)
# 恢复任务进度
def restore_progress(id_task):
    # 当前任务仍在进行或待处理任务列表中存在该任务时，等待0.1秒
    while id_task == current_task or id_task in pending_tasks:
        time.sleep(0.1)

    # 从记录的结果中查找与给定任务ID匹配的结果，如果找到则返回该结果
    res = next(iter([x[1] for x in recorded_results if id_task == x[0]]), None)
    if res is not None:
        return res

    # 如果未找到匹配结果，则返回更新的结果和错误消息
    return gr.update(), gr.update(), gr.update(), f"Couldn't restore progress for {id_task}: results either have been discarded or never were obtained"
```