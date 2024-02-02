# `so-vits-svc\diffusion\uni_pc.py`

```py
import math  # 导入数学库
import torch  # 导入PyTorch库


class NoiseScheduleVP:  # 定义NoiseScheduleVP类
    def marginal_log_mean_coeff(self, t):  # 定义计算给定连续时间标签t在[0, T]范围内的log(alpha_t)的方法
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':  # 如果调度方式为'discrete'
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))  # 返回插值函数计算的结果
        elif self.schedule == 'linear':  # 如果调度方式为'linear'
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0  # 返回线性计算的结果
        elif self.schedule == 'cosine':  # 如果调度方式为'cosine'
            def log_alpha_fn(s):  # 定义log_alpha_fn函数
                return torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))  # 返回cosine函数的对数值
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0  # 计算log_alpha_t的值
            return log_alpha_t  # 返回log_alpha_t的值

    def marginal_alpha(self, t):  # 定义计算给定连续时间标签t在[0, T]范围内的alpha_t的方法
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))  # 返回exp函数计算的结果

    def marginal_std(self, t):  # 定义计算给定连续时间标签t在[0, T]范围内的sigma_t的方法
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))  # 返回计算结果

    def marginal_lambda(self, t):  # 定义计算给定连续时间标签t在[0, T]范围内的lambda_t的方法
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)  # 计算log_mean_coeff的值
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))  # 计算log_std的值
        return log_mean_coeff - log_std  # 返回计算结果
    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        # 如果调度方式为线性
        if self.schedule == 'linear':
            # 计算临时变量 tmp
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            # 计算 Delta
            Delta = self.beta_0**2 + tmp
            # 返回 t
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        # 如果调度方式为离散
        elif self.schedule == 'discrete':
            # 计算 log_alpha
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            # 插值计算 t
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            # 计算 log_alpha
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            # 定义 t_fn 函数
            def t_fn(log_alpha_t):
                return torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            # 计算 t
            t = t_fn(log_alpha)
            return t
def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        # 根据噪声时间表的类型，将连续时间 t_continuous 转换为模型输入时间 t_input
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        # 获取模型输入时间
        t_input = get_model_input_time(t_continuous)
        # 根据条件调用模型，获取输出
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        # 根据模型类型返回不同的结果
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output
    # 定义一个函数，用于计算分类器的梯度，即 nabla_{x} log p_t(cond | x_t)
    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        # 启用梯度计算
        with torch.enable_grad():
            # 将输入的张量 x 分离出来，并设置其需要梯度计算
            x_in = x.detach().requires_grad_(True)
            # 使用分类器函数计算对数概率
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            # 计算 log_prob 的梯度，并返回
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    # 定义一个模型函数，用于 DPM-Solver 中的噪声预测模型
    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        # 如果指导类型是 "uncond"，则直接返回噪声预测函数的结果
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        # 如果指导类型是 "classifier"，则进行以下操作
        elif guidance_type == "classifier":
            # 断言分类器函数不为空
            assert classifier_fn is not None
            # 获取模型输入的时间
            t_input = get_model_input_time(t_continuous)
            # 计算条件梯度
            cond_grad = cond_grad_fn(x, t_input)
            # 获取时间 t_continuous 对应的噪声标准差
            sigma_t = noise_schedule.marginal_std(t_continuous)
            # 计算噪声预测函数的结果
            noise = noise_pred_fn(x, t_continuous)
            # 返回噪声减去指导比例乘以标准差和条件梯度的结果
            return noise - guidance_scale * sigma_t * cond_grad
        # 如果指导类型是 "classifier-free"，则进行以下操作
        elif guidance_type == "classifier-free":
            # 如果指导比例为 1 或者无条件条件为空，则进行以下操作
            if guidance_scale == 1. or unconditional_condition is None:
                # 返回噪声预测函数的结果
                return noise_pred_fn(x, t_continuous, cond=condition)
            # 否则，进行以下操作
            else:
                # 复制两份输入张量 x
                x_in = torch.cat([x] * 2)
                # 复制两份时间 t_continuous
                t_in = torch.cat([t_continuous] * 2)
                # 复制无条件条件和条件
                c_in = torch.cat([unconditional_condition, condition])
                # 使用噪声预测函数计算结果，并分割成两部分
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                # 返回无条件噪声加上指导比例乘以有条件噪声和无条件噪声的差
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    # 断言模型类型为 "noise"、"x_start" 或 "v"
    assert model_type in ["noise", "x_start", "v"]
    # 断言指导类型为 "uncond"、"classifier" 或 "classifier-free"
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    # 返回模型函数
    return model_fn
class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        variant='bh1'
    ):
        """Construct a UniPC. 

        We support both data_prediction and noise_prediction.
        """
        # 初始化 UniPC 类
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0]))
        # 设置模型函数
        self.noise_schedule = noise_schedule
        # 设置噪声调度
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        # 断言算法类型为数据预测或噪声预测

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        # 如果修正 x0 函数为动态阈值，则设置为动态阈值函数，否则设置为传入的修正 x0 函数

        self.correcting_xt_fn = correcting_xt_fn
        # 设置修正 xt 函数
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        # 设置动态阈值比率
        self.thresholding_max_val = thresholding_max_val
        # 设置阈值最大值

        self.variant = variant
        # 设置变体
        self.predict_x0 = algorithm_type == "data_prediction"
        # 如果算法类型为数据预测，则设置预测 x0 为真

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method. 
        """
        # 动态阈值方法
        dims = x0.dim()
        # 获取 x0 的维度
        p = self.dynamic_thresholding_ratio
        # 获取动态阈值比率
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        # 计算 x0 绝对值的分位数
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        # 将 s 扩展为与 x0 相同的维度
        x0 = torch.clamp(x0, -s, s) / s
        # 对 x0 进行截断和归一化
        return x0
        # 返回处理后的 x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        # 返回噪声预测模型
        return self.model(x, t)
        # 调用模型函数进行预测
    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        # 计算噪声预测
        noise = self.noise_prediction_fn(x, t)
        # 计算噪声的边际 alpha 和标准差 sigma
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        # 计算修正后的数据预测模型
        x0 = (x - sigma_t * noise) / alpha_t
        # 如果存在修正函数，则对修正后的数据进行修正
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        # 返回修正后的数据预测模型
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        # 如果需要预测 x0，则返回数据预测模型，否则返回噪声预测模型
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        # 根据不同的跳跃类型计算中间时间步
        if skip_type == 'logSNR':
            # 计算 lambda_T 和 lambda_0
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            # 计算 logSNR 步长
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            # 返回逆转的 lambda 值
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            # 返回均匀分布的时间步
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            # 计算二次时间步
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            # 抛出不支持的跳跃类型异常
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))
    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        # 根据给定的步数和阶数计算每个步骤的顺序
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # 为了复现 DPM-Solver 论文中的结果
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        # 在最后一步去噪，相当于通过一阶离散化解决从 lambda_s 到无穷大的 ODE
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, **kwargs):
        # 如果 t 的维度为 0，则将其视为一维
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
    # 定义一个方法，用于对输入数据进行采样
    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, atol=0.0078, rtol=0.05, return_intermediate=False,
# 定义一个分段线性函数 y = f(x)，使用 xp 和 yp 作为关键点
# 以可微分的方式实现 f(x)（适用于 autograd）
# 函数 f(x) 对所有 x 轴都是明确定义的（对于超出 xp 范围的 x，我们使用 xp 的最外层点来定义线性函数）

def interpolate_fn(x, xp, yp):
    # 获取 x 的批处理大小 N 和关键点数量 K
    N, K = x.shape[0], xp.shape[1]
    # 将 x 和 xp 连接起来，形成一个新的张量 all_x
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    # 对 all_x 进行排序，并记录排序后的索引
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    # 获取 x 的索引
    x_idx = torch.argmin(x_indices, dim=2)
    # 计算起始索引
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    # 计算结束索引
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    # 获取起始 x 值和结束 x 值
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    # 计算起始索引2
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    # 扩展 yp，使其与 start_idx2 的形状相匹配
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    # 获取起始 y 值
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    # 从 y_positions_expanded 中按照 index=(start_idx2 + 1).unsqueeze(2) 的值，沿着 dim=2 维度进行聚合操作，得到 end_y
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # 计算候选值 cand，根据线性插值公式 (x - start_x) * (end_y - start_y) / (end_x - start_x) + start_y
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    # 返回候选值
    return cand
# 将张量 `v` 扩展到维度 `dims`

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].  # 输入参数 `v` 是一个形状为 [N] 的 PyTorch 张量
        `dim`: a `int`.  # 输入参数 `dim` 是一个整数
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.  # 返回一个形状为 [N, 1, 1, ..., 1] 的 PyTorch 张量，总维度为 `dims`
    """
    return v[(...,) + (None,)*(dims - 1)]  # 返回扩展后的张量
```