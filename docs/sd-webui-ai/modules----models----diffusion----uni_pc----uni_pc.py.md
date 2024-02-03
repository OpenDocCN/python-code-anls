# `stable-diffusion-webui\modules\models\diffusion\uni_pc\uni_pc.py`

```
import torch
import math
import tqdm

# 定义噪声调度器类
class NoiseScheduleVP:
    # 计算给定连续时间标签 t 在 [0, T] 区间内的 log(alpha_t)
    def marginal_log_mean_coeff(self, t):
        # 如果调度方式为离散
        if self.schedule == 'discrete':
            # 调用插值函数计算 log(alpha_t)
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        # 如果调度方式为线性
        elif self.schedule == 'linear':
            # 计算 log(alpha_t) 的线性函数
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        # 如果调度方式为余弦
        elif self.schedule == 'cosine':
            # 定义余弦函数
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            # 计算 log(alpha_t) 的余弦函数
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    # 计算给定连续时间标签 t 在 [0, T] 区间内的 alpha_t
    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    # 计算给定连续时间标签 t 在 [0, T] 区间内的 sigma_t
    def marginal_std(self, t):
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    # 计算给定连续时间标签 t 在 [0, T] 区间内的 lambda_t = log(alpha_t) - log(sigma_t)
    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std
    # 计算给定半对数信噪比 lambda_t 的连续时间标签 t，范围在 [0, T] 之间
    def inverse_lambda(self, lamb):
        # 如果调度方式为线性
        if self.schedule == 'linear':
            # 计算临时变量 tmp
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            # 计算 Delta
            Delta = self.beta_0**2 + tmp
            # 返回计算结果
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        # 如果调度方式为离散
        elif self.schedule == 'discrete':
            # 计算 log_alpha
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            # 插值计算 t
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            # 返回计算结果
            return t.reshape((-1,))
        else:
            # 计算 log_alpha
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            # 定义 lambda 函数 t_fn
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            # 计算 t
            t = t_fn(log_alpha)
            # 返回计算结果
            return t
# 创建一个模型包装器函数，用于噪声预测模型
def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs=None,
    guidance_type="uncond",
    #condition=None,
    #unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs=None,
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```
    # 我们支持三种类型的 DPM 引导采样，通过设置 `guidance_type`：
        # 1. "uncond": 无条件 DPM 采样。
            # 输入 `model` 的格式如下：
            # ``
                # model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            # ``

        # 2. "classifier": 通过 DPM 和另一个分类器进行分类器引导采样 [3]。
            # 输入 `model` 的格式如下：
            # ``
                # model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            # ``

            # 输入 `classifier_fn` 的格式如下：
            # ``
                # classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            # ``

            # [3] P. Dhariwal 和 A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                # 在 Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794 中。

        # 3. "classifier-free": 通过条件 DPM 进行无分类器引导采样。
            # 输入 `model` 的格式如下：
            # ``
                # model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            # ``
            # 如果 cond == `unconditional_condition`，则模型输出为无条件 DPM 输出。

            # [4] Ho, Jonathan, 和 Tim Salimans. "Classifier-free diffusion guidance."
                # arXiv 预印本 arXiv:2207.12598 (2022)。


    # `t_input` 是模型的时间标签，可以是离散时间标签（即 0 到 999）
    # 或连续时间标签（即 epsilon 到 T）。

    # 我们包装模型函数，只接受 `x` 和 `t_continuous` 作为输入，并输出预测的噪声：
    # ``
        # def model_fn(x, t_continuous) -> noise:
            # t_input = get_model_input_time(t_continuous)
            # return noise_pred(model, x, t_input, **model_kwargs)
    # ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    # Initialize model_kwargs and classifier_kwargs if not provided
    model_kwargs = model_kwargs or {}
    classifier_kwargs = classifier_kwargs or {}

    # Define a function to convert continuous-time to model input time based on noise schedule
    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        # Check if noise schedule is discrete or continuous
        if noise_schedule.schedule == 'discrete':
            # Convert continuous-time to model input time for discrete-time DPMs
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            # Use continuous-time directly for continuous-time DPMs
            return t_continuous
    # 定义一个函数，用于生成噪声预测
    def noise_pred_fn(x, t_continuous, cond=None):
        # 如果时间连续值的形状为1，则将其扩展为与输入数据相同的形状
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        # 获取模型输入的时间信息
        t_input = get_model_input_time(t_continuous)
        # 根据条件选择模型的输出
        if cond is None:
            output = model(x, t_input, None, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        # 根据模型类型返回不同的输出
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            # 计算 alpha_t 和 sigma_t
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            # 计算 alpha_t 和 sigma_t
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            # 计算 sigma_t
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    # 定义一个函数，用于计算条件梯度
    def cond_grad_fn(x, t_input, condition):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        # 启用梯度计算
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # 计算分类器的对数概率
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            # 返回对输入数据的梯度
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    # 断言模型类型和引导类型的取值范围
    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    # 返回模型函数
    return model_fn
class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0=True,
        thresholding=False,
        max_val=1.,
        variant='bh1',
        condition=None,
        unconditional_condition=None,
        before_sample=None,
        after_sample=None,
        after_update=None
    ):
        """Construct a UniPC.

        We support both data_prediction and noise_prediction.
        """
        # 初始化 UniPC 类
        self.model_fn_ = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.condition = condition
        self.unconditional_condition = unconditional_condition
        self.before_sample = before_sample
        self.after_sample = after_sample
        self.after_update = after_update

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        # 动态阈值方法
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model(self, x, t):
        cond = self.condition
        uncond = self.unconditional_condition
        if self.before_sample is not None:
            x, t, cond, uncond = self.before_sample(x, t, cond, uncond)
        res = self.model_fn_(x, t, cond, uncond)
        if self.after_sample is not None:
            x, t, cond, uncond, res = self.after_sample(x, t, cond, uncond, res)

        if isinstance(res, tuple):
            # (None, pred_x0)
            res = res[1]

        return res

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        # 返回噪声预测模型
        return self.model(x, t)
    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        # 使用噪声预测函数获取噪声
        noise = self.noise_prediction_fn(x, t)
        # 获取输入张量的维度
        dims = x.dim()
        # 获取时间步 t 对应的 alpha 和 sigma
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        # 计算 x0
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        # 如果开启了阈值处理
        if self.thresholding:
            # 设置超参数 p
            p = 0.995   # A hyperparameter in the paper of "Imagen" [1].
            # 计算阈值 s
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims)
            # 对 x0 进行阈值处理
            x0 = torch.clamp(x0, -s, s) / s
        # 返回处理后的 x0
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        # 如果需要预测 x0，则调用 data_prediction_fn 函数
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        # 否则调用 noise_prediction_fn 函数
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        # 根据不同的 skip_type 计算中间时间步
        if skip_type == 'logSNR':
            # 获取 t_T 和 t_0 对应的 lambda
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            # 在 logSNR 空间上均匀采样 N 个点
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            # 将 logSNR 转换为 lambda
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            # 在 t_T 和 t_0 之间均匀采样 N 个点
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            # 在 t_T^(1/2) 和 t_0^(1/2) 之间按二次函数采样 N 个点
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            # 抛出异常，不支持的 skip_type
            raise ValueError(f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'")
    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        # 根据给定的步数和顺序确定每个步骤的顺序
        if order == 3:
            K = steps // 3 + 1
            # 如果步数可以被3整除
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            # 如果步数除以3余1
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            # 如果步数除以3余2
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            # 如果顺序为2
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            # 如果顺序为1
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # 为了复现DPM-Solver论文中的结果
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        # 在最后一步去噪，相当于通过一阶离散化从lambda_s到无穷解ODE
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, **kwargs):
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
    # 定义一个方法sample，接受多个参数
    # x: 输入数据
    # steps: 迭代步数，默认为20
    # t_start: 开始时间，默认为None
    # t_end: 结束时间，默认为None
    # order: 阶数，默认为3
    # skip_type: 跳过类型，默认为'time_uniform'
    # method: 方法类型，默认为'singlestep'
    # lower_order_final: 是否使用较低阶数，默认为True
    # denoise_to_zero: 是否去噪到零，默认为False
    # solver_type: 求解器类型，默认为'dpm_solver'
    # atol: 绝对误差容限，默认为0.0078
    # rtol: 相对误差容限，默认为0.05
    # corrector: 是否进行校正，默认为False
# 定义一个分段线性函数 y = f(x)，使用 xp 和 yp 作为关键点。
# 我们以可微分的方式实现 f(x)（适用于 autograd）。
# 函数 f(x) 在整个 x 轴上都是明确定义的。（对于超出 xp 范围的 x，我们使用 xp 的最外侧点来定义线性函数。）

def interpolate_fn(x, xp, yp):
    # 获取 x 的形状 [N, C]，其中 N 是批量大小，C 是通道数（我们在 DPM-Solver 中使用 C = 1）。
    N, K = x.shape[0], xp.shape[1]
    # 将 x 和 xp 连接起来，形成一个新的张量 all_x，形状为 [N, C, K+1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    # 对 all_x 进行排序，并记录排序后的索引
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    # 找到 x 在 xp 中的索引
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
    # 获取起始点的 x 值
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    # 获取结束点的 x 值
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    # 计算起始索引2
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    # 扩展 yp 的维度，使其与 start_idx2 的维度相匹配
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    # 获取起始点的 y 值
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    # 从 y_positions_expanded 中根据 start_idx2 + 1 的索引获取对应的值，然后挤压成一维张量
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # 计算候选值 cand，根据线性插值公式计算出当前 x 对应的 y 值
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    # 返回计算出的候选值
    return cand
# 将张量 `v` 扩展到维度 `dims`。

def expand_dims(v, dims):
    # `v`: 一个形状为 [N] 的 PyTorch 张量。
    # `dim`: 一个整数。
    # 返回一个形状为 [N, 1, 1, ..., 1] 的 PyTorch 张量，总维度为 `dims`。
    return v[(...,) + (None,)*(dims - 1)]
```