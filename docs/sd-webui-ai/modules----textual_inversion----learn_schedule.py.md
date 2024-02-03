# `stable-diffusion-webui\modules\textual_inversion\learn_schedule.py`

```py
import tqdm

# 定义一个学习率调度迭代器类
class LearnScheduleIterator:
    def __init__(self, learn_rate, max_steps, cur_step=0):
        """
        specify learn_rate as "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000
        """
        
        # 将学习率字符串按逗号分隔成多个学习率步长对
        pairs = learn_rate.split(',')
        self.rates = []
        self.it = 0
        self.maxit = 0
        try:
            # 遍历每个学习率步长对
            for pair in pairs:
                if not pair.strip():
                    continue
                tmp = pair.split(':')
                # 如果步长对包含两个元素
                if len(tmp) == 2:
                    step = int(tmp[1])
                    # 如果步长大于当前步数，则添加到学习率列表中
                    if step > cur_step:
                        self.rates.append((float(tmp[0]), min(step, max_steps)))
                        self.maxit += 1
                        if step > max_steps:
                            return
                    # 如果步长为-1，则添加到学习率列表中
                    elif step == -1:
                        self.rates.append((float(tmp[0]), max_steps))
                        self.maxit += 1
                        return
                # 如果步长对只包含一个元素
                else:
                    self.rates.append((float(tmp[0]), max_steps))
                    self.maxit += 1
                    return
            # 确保学习率列表不为空
            assert self.rates
        except (ValueError, AssertionError) as e:
            raise Exception('Invalid learning rate schedule. It should be a number or, for example, like "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000.') from e

    # 定义迭代器的迭代方法
    def __iter__(self):
        return self

    # 定义迭代器的下一个方法
    def __next__(self):
        if self.it < self.maxit:
            self.it += 1
            return self.rates[self.it - 1]
        else:
            raise StopIteration

# 定义学习率调度器类
class LearnRateScheduler:
    # 初始化学习率调度器对象，设置学习率、最大步数、当前步数、是否显示详细信息等属性
    def __init__(self, learn_rate, max_steps, cur_step=0, verbose=True):
        # 创建学习率调度器迭代器对象
        self.schedules = LearnScheduleIterator(learn_rate, max_steps, cur_step)
        # 获取下一个学习率和结束步数
        (self.learn_rate,  self.end_step) = next(self.schedules)
        # 设置是否显示详细信息
        self.verbose = verbose

        # 如果需要显示详细信息，则打印当前训练速率和结束步数
        if self.verbose:
            print(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        # 初始化训练是否完成的标志
        self.finished = False

    # 更新学习率和结束步数，并检查是否训练完成
    def step(self, step_number):
        # 如果当前步数小于结束步数，则返回 False
        if step_number < self.end_step:
            return False

        # 尝试获取下一个学习率和结束步数
        try:
            (self.learn_rate, self.end_step) = next(self.schedules)
        # 如果迭代结束，则将训练完成标志设置为 True，并返回 False
        except StopIteration:
            self.finished = True
            return False
        # 返回 True
        return True

    # 应用学习率到优化器的参数组中
    def apply(self, optimizer, step_number):
        # 如果更新学习率和结束步数失败，则直接返回
        if not self.step(step_number):
            return

        # 如果需要显示详细信息，则打印当前训练速率和结束步数
        if self.verbose:
            tqdm.tqdm.write(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        # 遍历优化器的参数组，将学习率设置为当前学习率
        for pg in optimizer.param_groups:
            pg['lr'] = self.learn_rate
```