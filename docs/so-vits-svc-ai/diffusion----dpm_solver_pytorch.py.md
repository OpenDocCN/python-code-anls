# `so-vits-svc\diffusion\dpm_solver_pytorch.py`

```
import torch

class NoiseScheduleVP:
    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        # 计算标准差
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        # 计算 lambda
        lambs = log_alphas - log_sigmas  
        # 寻找超出阈值的索引
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        # 如果存在超出阈值的索引，则截取 log_alphas
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            # 根据离散时间标签 t 计算 log(alpha_t)
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            # 根据线性时间标签 t 计算 log(alpha_t)
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        # 计算 alpha_t
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        # 计算标准差 sigma_t
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        # 计算 lambda_t
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std
    # 计算给定半对数信噪比 lambda_t 的连续时间标签 t，范围在 [0, T] 之间
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
    """
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

            # [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                # in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        # 3. "classifier-free": 通过条件 DPM 进行无分类器引导采样。
            # 输入 `model` 的格式如下：
            # ``
                # model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            # `` 
            # 如果 cond == `unconditional_condition`，则模型输出为无条件 DPM 输出。

            # [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                # arXiv preprint arXiv:2207.12598 (2022).
        

    # `t_input` 是模型的时间标签，可以是离散时间标签（即 0 到 999）或连续时间标签（即 epsilon 到 T）。

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

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        # 根据噪声调度的类型，将连续时间转换为模型输入时间
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous
    # 定义一个函数，用于对噪声模型进行预测
    def noise_pred_fn(x, t_continuous, cond=None):
        # 根据连续时间获取模型输入时间
        t_input = get_model_input_time(t_continuous)
        # 如果条件为空，则使用模型对输入进行预测
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        # 如果有条件，则使用条件和模型对输入进行预测
        else:
            output = model(x, t_input, cond, **model_kwargs)
        # 如果模型类型是噪声，则直接返回输出
        if model_type == "noise":
            return output
        # 如果模型类型是 x_start，则根据噪声调度计划对输出进行处理
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(sigma_t, x.dim())
        # 如果模型类型是 v，则根据噪声调度计划对输出进行处理
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return expand_dims(alpha_t, x.dim()) * output + expand_dims(sigma_t, x.dim()) * x
        # 如果模型类型是 score，则根据噪声调度计划对输出进行处理
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    # 定义一个函数，用于计算分类器的梯度，即 nabla_{x} log p_t(cond | x_t)
    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        # 启用梯度计算
        with torch.enable_grad():
            # 将输入张量分离出来，并设置其需要梯度计算
            x_in = x.detach().requires_grad_(True)
            # 计算分类器的对数概率
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            # 返回对输入张量的梯度
            return torch.autograd.grad(log_prob.sum(), x_in)[0]
    # 定义一个模型函数，用于 DPM-Solver 的噪声预测模型
    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        # 如果指导类型是“uncond”，则返回噪声预测函数对应的结果
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        # 如果指导类型是“classifier”，则进行条件判断
        elif guidance_type == "classifier":
            # 断言分类器函数不为空
            assert classifier_fn is not None
            # 获取模型输入时间
            t_input = get_model_input_time(t_continuous)
            # 计算条件梯度
            cond_grad = cond_grad_fn(x, t_input)
            # 计算噪声标准差
            sigma_t = noise_schedule.marginal_std(t_continuous)
            # 计算噪声
            noise = noise_pred_fn(x, t_continuous)
            # 返回噪声减去指导比例乘以扩展后的条件梯度和噪声标准差
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        # 如果指导类型是“classifier-free”，则进行条件判断
        elif guidance_type == "classifier-free":
            # 如果指导比例为1或无条件条件为空，则返回噪声预测函数对应的结果
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            # 否则进行以下操作
            else:
                # 复制输入的 x 和 t_continuous
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                # 对噪声预测函数的结果进行分块
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                # 返回无条件噪声加上指导比例乘以噪声减去无条件噪声
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    # 断言模型类型为噪声、起始位置、速度或分数
    assert model_type in ["noise", "x_start", "v", "score"]
    # 断言指导类型为无条件、分类器或无分类器
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    # 返回模型函数
    return model_fn
class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    # 初始化方法，设置模型函数、噪声调度、算法类型、修正初始值函数、修正终值函数、阈值最大值、动态阈值比率
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        # 动态阈值方法，计算 x0 的维度
        dims = x0.dim()
        # 获取 x0 的绝对值并按行计算 p 分位数
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        # 将 s 扩展为与 x0 相同的维度，并与阈值最大值相比较取较大值
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        # 对 x0 进行截断并归一化
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        # 返回噪声预测模型
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        # 返回数据预测模型（带有修正器）
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        # 将模型转换为噪声预测模型或数据预测模型
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)
    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            # 计算起始时间和结束时间对应的噪声强度
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            # 在噪声强度范围内均匀取N+1个点
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            # 在起始时间和结束时间之间均匀取N+1个点
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            # 在起始时间和结束时间的t_order次方根范围内均匀取N+1个点，再取t_order次方
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        # 在最终步骤进行去噪，相当于通过一阶离散化解决从lambda_s到无穷大的ODE
        return self.data_prediction_fn(x, s)
    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 计算时间点 s 和 t 对应的边际 lambda 值
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        # 计算时间间隔 h
        h = lambda_t - lambda_s
        # 计算时间点 s 和 t 对应的边际对数均值系数
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        # 计算时间点 s 和 t 对应的边际标准差
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        # 计算时间点 t 对应的均值系数的指数
        alpha_t = torch.exp(log_alpha_t)

        # 如果算法类型为 "dpmsolver++"
        if self.algorithm_type == "dpmsolver++":
            # 计算 phi_1
            phi_1 = torch.expm1(-h)
            # 如果模型函数 model_s 为空，则通过 x 和 s 计算模型函数
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 计算时间点 t 对应的近似解 x_t
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            # 如果需要返回中间结果，则返回 x_t 和 model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            # 否则只返回 x_t
            else:
                return x_t
        # 如果算法类型不是 "dpmsolver++"
        else:
            # 计算 phi_1
            phi_1 = torch.expm1(h)
            # 如果模型函数 model_s 为空，则通过 x 和 s 计算模型函数
            if model_s is None:
                model_s = self.model_fn(x, s)
            # 计算时间点 t 对应的近似解 x_t
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            # 如果需要返回中间结果，则返回 x_t 和 model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            # 否则只返回 x_t
            else:
                return x_t
    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 如果 order 为 1，则调用一阶 DPM-Solver 更新函数
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        # 如果 order 为 2，则调用二阶 DPM-Solver 更新函数
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        # 如果 order 为 3，则调用三阶 DPM-Solver 更新函数
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        # 如果 order 不是 1、2、3 中的任何一个，则抛出数值错误
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        # 如果 order 为 1，则调用 dpm_solver_first_update 方法
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        # 如果 order 为 2，则调用 multistep_dpm_solver_second_update 方法
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        # 如果 order 为 3，则调用 multistep_dpm_solver_third_update 方法
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        # 如果 order 不在 1、2、3 中，则抛出数值错误
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))
    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        # 获取 alpha_t 和 sigma_t
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        # 如果没有提供噪声数据，则生成一个符合要求的噪声数据
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        # 重塑输入数据 x 的形状
        x = x.reshape((-1, *x.shape))
        # 计算加噪后的输入数据 xt
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        # 如果 t 的长度为 1，则压缩维度并返回结果
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        # 如果未指定 t_start，则使用默认值
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        # 如果未指定 t_end，则使用默认值
        t_T = self.noise_schedule.T if t_end is None else t_end
        # 断言 t_0 和 t_T 大于 0
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        # 调用 sample 方法进行采样
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)
    # 定义一个方法，用于对输入数据进行采样
    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
# 定义一个插值函数，用于计算分段线性函数 y = f(x)，使用 xp 和 yp 作为关键点
# 以可微分的方式实现 f(x)（适用于 autograd），f(x) 对所有 x 轴都是明确定义的
# 对于超出 xp 范围的 x，使用 xp 的最外侧点来定义线性函数

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
    # 扩展 yp，以便与 start_idx2 对应
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    # 获取起始 y 值
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    # 从 y_positions_expanded 中按照 index=(start_idx2 + 1).unsqueeze(2) 的值，沿着 dim=2 维度进行聚合操作，得到 end_y
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # 计算候选值，根据线性插值公式计算出候选值
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