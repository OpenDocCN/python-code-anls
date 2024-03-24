# `.\lucidrains\big-sleep\test\multi_prompt_minmax.py`

```
# 导入所需的库
import time
import shutil
import torch
from big_sleep import Imagine

# 初始化终止标志
terminate = False

# 信号处理函数，设置终止标志为True
def signal_handling(signum,frame):
    global terminate
    terminate = True

# 设定尝试次数
num_attempts = 4
# 循环尝试生成图像
for attempt in range(num_attempts):
    # 创建Imagine对象，用于生成图像
    dream = Imagine(
        text = "an armchair in the form of pikachu\\an armchair imitating pikachu\\abstract",
        text_min = "blur\\zoom",
        lr = 7e-2,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 5,
        iterations = 50,
        save_progress = False,
        bilinear = False,
        open_folder = False,
        seed = None,
        torch_deterministic = False,
        max_classes = 20,
        class_temperature = 2.,
        save_date_time = False,
        save_best = True,
        experimental_resample = True,
        ema_decay = 0.99
    )
    # 生成图像
    dream()
    # 复制生成的最佳图像
    shutil.copy(dream.textpath + ".best.png", f"{attempt}.png")
    try:
        # 等待2秒
        time.sleep(2)
        # 删除dream对象
        del dream
        # 再次等待2秒
        time.sleep(2)
        # 清空GPU缓存
        torch.cuda.empty_cache()
    except Exception:
        # 出现异常时，仅清空GPU缓存
        torch.cuda.empty_cache()
```