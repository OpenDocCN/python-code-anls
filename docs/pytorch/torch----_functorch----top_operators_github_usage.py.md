# `.\pytorch\torch\_functorch\top_operators_github_usage.py`

```
# 忽略 mypy 的错误，用于类型检查工具的声明
# 多行字符串，包含了一个 Google 表格的链接，用于参考和同步列表
"""
From https://docs.google.com/spreadsheets/d/12R3nCOLskxPYjjiNkdqy4OdQ65eQp_htebXGODsjSeA/edit#gid=0
Try to keep this list in sync with that.
"""
# 导入 operator 模块，用于操作符函数的标准集合
import operator

# 定义一个列表 top_torch，包含了元组，每个元组表示了一个 Torch 库的顶级函数或方法名及其出现频率
top_torch = [
    ("t", 6837449),        # 't' 出现了 6837449 次
    ("tensor", 585786),    # 'tensor' 出现了 585786 次
    ("mode", 462182),      # 'mode' 出现了 462182 次
    ("cat", 394818),       # 'cat' 出现了 394818 次
    ("max", 368038),       # 'max' 出现了 368038 次
    ("zeros", 329495),     # 'zeros' 出现了 329495 次
    ("load", 327756),      # 'load' 出现了 327756 次
    ("no_grad", 294694),   # 'no_grad' 出现了 294694 次
    ("save", 265130),      # 'save' 出现了 265130 次
    ("from_numpy", 243063),# 'from_numpy' 出现了 243063 次
    ("manual_seed", 165044),  # 'manual_seed' 出现了 165044 次
    ("ones", 153696),      # 'ones' 出现了 153696 次
    ("randn", 150796),     # 'randn' 出现了 150796 次
    ("stack", 133358),     # 'stack' 出现了 133358 次
    ("sum", 130772),       # 'sum' 出现了 130772 次
    ("arange", 98087),     # 'arange' 出现了 98087 次
    ("rand", 94715),       # 'rand' 出现了 94715 次
    ("mean", 88546),       # 'mean' 出现了 88546 次
    ("exp", 73883),        # 'exp' 出现了 73883 次
    ("zeros_like", 72831), # 'zeros_like' 出现了 72831 次
    ("min", 72248),        # 'min' 出现了 72248 次
    ("sigmoid", 66798),    # 'sigmoid' 出现了 66798 次
    ("log", 62135),        # 'log' 出现了 62135 次
    ("matmul", 47811),     # 'matmul' 出现了 47811 次
    ("clamp", 45304),      # 'clamp' 出现了 45304 次
    ("sqrt", 44911),       # 'sqrt' 出现了 44911 次
    ("abs", 43535),        # 'abs' 出现了 43535 次
    ("tanh", 42793),       # 'tanh' 出现了 42793 次
    ("empty", 40311),      # 'empty' 出现了 40311 次
    ("argmax", 38435),     # 'argmax' 出现了 38435 次
    ("bmm", 33984),        # 'bmm' 出现了 33984 次
    ("pow", 33571),        # 'pow' 出现了 33571 次
    ("norm", 31125),       # 'norm' 出现了 31125 次
    ("mm", 30995),         # 'mm' 出现了 30995 次
    ("is_tensor", 29546),  # 'is_tensor' 出现了 29546 次
    ("ones_like", 29512),  # 'ones_like' 出现了 29512 次
    ("nonzero", 28681),    # 'nonzero' 出现了 28681 次
    ("full", 28373),       # 'full' 出现了 28373 次
    ("unsqueeze", 27911),  # 'unsqueeze' 出现了 27911 次
    ("where", 26585),      # 'where' 出现了 26585 次
    ("randperm", 26450),   # 'randperm' 出现了 26450 次
    ("eye", 24342),        # 'eye' 出现了 24342 次
    ("mul", 23236),        # 'mul' 出现了 23236 次
    ("topk", 22537),       # 'topk' 出现了 22537 次
    ("as_tensor", 21967),  # 'as_tensor' 出现了 21967 次
    ("sort", 21412),       # 'sort' 出现了 21412 次
    ("squeeze", 20863),    # 'squeeze' 出现了 20863 次
    ("randint", 20771),    # 'randint' 出现了 20771 次
    ("linspace", 20041),   # 'linspace' 出现了 20041 次
    ("add", 19201),        # 'add' 出现了 19201 次
    ("transpose", 18663),  # 'transpose' 出现了 18663 次
    ("split", 18325),      # 'split' 出现了 18325 次
    ("gather", 17904),     # 'gather' 出现了 17904 次
    ("set_grad_enabled", 16013),  # 'set_grad_enabled' 出现了 16013 次
    ("sin", 15669),        # 'sin' 出现了 15669 次
    ("cos", 15562),        # 'cos' 出现了 15562 次
    ("div", 15513),        # 'div' 出现了 15513 次
    ("index_select", 14866),  # 'index_select' 出现了 14866 次
    ("multinomial", 14331), # 'multinomial' 出现了 14331 次
    ("flatten", 14267),    # 'flatten' 出现了 14267 次
    ("isnan", 14170),      # 'isnan' 出现了 14170 次
    ("randn_like", 13096), # 'randn_like' 出现了 13096 次
    ("eq", 12680),         # 'eq' 出现了 12680 次
    ("einsum", 12480),     # 'einsum' 出现了 12480 次
    ("round", 12367),      # 'round' 出现了 12367 次
    ("floor", 11628),      # 'floor' 出现了 11628 次
    ("allclose", 11000),   # 'allclose' 出现了 11000 次
    ("reshape", 10605),    # 'reshape' 出现了 10605 次
    ("diag", 10167),       # 'diag' 出现了 10167 次
    ("chunk", 9581),       # 'chunk' 出现了 9581 次
    ("std", 9379),         # 'std' 出现了 9379 次
    ("set_default_tensor_type", 9281),  # 'set_default_tensor_type' 出现了 9281 次
    ("triu", 8559),        # 'triu' 出现了 8559 次
    ("meshgrid", 8292),    # 'meshgrid' 出现了 8292 次
    ("set_num_threads", 8126),  # 'set_num_threads' 出现了 8126 次
    ("unique
    # 导入整个 torch 库
    import torch
    
    # 创建一个稀疏 COO 张量
    sparse_coo_tensor = torch.sparse_coo_tensor
    
    # 计算以 10 为底的对数
    log10 = torch.log10
    
    # 返回张量中指定维度上第 k 个最小值及其对应的索引
    kthvalue = torch.kthvalue
    
    # 设置随机数生成器的状态
    set_rng_state = torch.set_rng_state
    
    # 获取当前随机数生成器的状态
    get_rng_state = torch.get_rng_state
    
    # 获取默认的张量数据类型
    get_default_dtype = torch.get_default_dtype
    
    # 计算矩阵的行列式
    det = torch.det
    
    # 对矩阵进行 QR 分解
    qr = torch.qr
    
    # 计算张量的直方图
    histc = torch.histc
    
    # 对对称矩阵进行特征值分解
    symeig = torch.symeig
    
    # 计算张量的迹
    trace = torch.trace
    
    # 计算张量的中位数
    median = torch.median
    
    # 执行张量的 addcmul 运算
    addcmul = torch.addcmul
    
    # 计算张量元素的余数
    remainder = torch.remainder
    
    # 对两个批量的矩阵进行元素乘法再加到一个批量的矩阵中
    baddbmm = torch.baddbmm
    
    # 计算张量元素的 lgamma 函数值
    lgamma = torch.lgamma
    
    # 对张量进行指定维度的重复插值
    repeat_interleave = torch.repeat_interleave
    
    # 计算张量元素的 fmod 函数值
    fmod = torch.fmod
    
    # 计算张量元素的倒数
    reciprocal = torch.reciprocal
    
    # 计算张量元素的正切值
    tan = torch.tan
    
    # 返回当前随机数生成器的初始种子
    initial_seed = torch.initial_seed
    
    # 返回张量中指定索引的元素
    take = torch.take
    
    # 对实部和虚部进行 STFT 转换
    stft = torch.stft
    
    # 获取当前线程的数目
    get_num_threads = torch.get_num_threads
    
    # 返回张量的实部
    real = torch.real
    
    # 对张量进行 Cholesky 分解
    cholesky = torch.cholesky
    
    # 对张量进行按元素量化
    quantize_per_tensor = torch.quantize_per_tensor
    
    # 将一维向量转换为对角矩阵
    diag_embed = torch.diag_embed
    
    # 在两个张量之间进行线性插值
    lerp = torch.lerp
    
    # 计算张量元素的反正弦值
    asin = torch.asin
    
    # 对方阵进行特征值分解
    eig = torch.eig
    
    # 计算张量元素的截断值
    trunc = torch.trunc
    
    # 返回矩阵的对角线元素
    diagonal = torch.diagonal
    
    # 计算张量元素的双曲余弦值
    cosh = torch.cosh
    
    # 对实数输入进行快速傅里叶变换
    rfft = torch.rfft
    
    # 计算张量元素的累积乘积
    cumprod = torch.cumprod
    
    # 对两个矩阵进行广播加法
    addr = torch.addr
    
    # 对张量进行循环移动
    roll = torch.roll
    
    # 返回张量的指定维度范围的视图
    narrow = torch.narrow
    
    # 计算张量元素的 digamma 函数值
    digamma = torch.digamma
    
    # 计算张量元素的平方
    square = torch.square
    
    # 计算张量元素的双曲正弦值
    sinh = torch.sinh
    
    # 返回在对数空间中均匀间隔的张量
    logspace = torch.logspace
    
    # 对张量进行广播操作
    broadcast_tensors = torch.broadcast_tensors
    
    # 对实数输入进行反快速傅里叶变换
    irfft = torch.irfft
    
    # 计算张量元素的分数部分
    frac = torch.frac
    
    # 返回汉宁窗口的张量表示
    hann_window = torch.hann_window
    
    # 解线性方程组
    solve = torch.solve
    
    # 计算矩阵的对数行列式
    logdet = torch.logdet
    
    # 计算张量元素的指数减一值
    expm1 = torch.expm1
    
    # 计算两组点之间的距离
    cdist = torch.cdist
    
    # 对张量进行矩阵-向量加法
    addmv = torch.addmv
    
    # 返回与输入张量具有相同形状和类型的随机整数张量
    randint_like = torch.randint_like
    
    # 计算张量的张量点积
    tensordot = torch.tensordot
    
    # 对复数输入进行快速傅里叶变换
    ifft = torch.ifft
    
    # 计算张量元素的真除法
    true_divide = torch.true_divide
    
    # 计算张量元素的反误差函数逆
    erfinv = torch.erfinv
    
    # 执行 addcdiv 运算
    addcdiv = torch.addcdiv
    
    # 对两个批量的矩阵进行逐元素乘法加到一个批量的矩阵中
    addbmm = torch.addbmm
    
    # 对张量进行重新标准化
    renorm = torch.renorm
    
    # 计算矩阵的伪逆
    pinverse = torch.pinverse
    
    # 判断两个张量的元素是否相近
    isclose = torch.isclose
    
    # 解三角矩阵的线性系统
    triangular_solve = torch.triangular_solve
    
    # 对矩阵进行旋转
    rot90 = torch.rot90
    
    # 计算逻辑非
    logical_not = torch.logical_not
    
    # 对矩阵进行 QR 分解并返回其 QR 分解的 QR 因子
    geqrf = torch.geqrf
    
    # 计算张量的实部和虚部的行列式对数
    slogdet = torch.slogdet
    
    # 对方阵进行 LU 分解
    lu = torch.lu
    
    # 返回巴特利特窗口的张量表示
    bartlett_window = torch.bartlett_window
    
    # 计算 Q 矩阵的 QR 分解
    orgqr = torch.orgqr
    
    # 计算 Q 矩阵的 RQ 分解
    ormqr = torch.ormqr
    
    # 判断张量是否为浮点类型
    is_floating_point = torch.is_floating_point
    
    # 返回扁平化的对角矩阵
    diagflat = torch.diagflat
    
    # 对方阵进行 Cholesky 解
    cholesky_solve = torch.cholesky_solve
    
    # 返回下三角矩阵的索引
    tril_indices = torch.tril_indices
    
    # 对一系列矩阵进行链式乘法
    chain_matmul = torch.chain_matmul
    
    # 返回上三角矩阵的索引
    triu_indices = torch.triu_indices
    
    # 计算复数的幅角
    angle = torch.angle
    
    # 返回服从泊松分布的张量
    poisson = torch.poisson
    
    # 计算矩阵的幂
    matrix_power = torch.matrix_power
    
    # 返回唯一连续值的张量
    unique_consecutive = torch.unique_consecutive
    
    # 对通道进行逐元素量化
    quantize_per_channel = torch.quantize_per_channel
    
    # 返回张量的标准差和均值
    std_mean = torch.std_mean
    
    # 返回巴特利特窗口的张量表示
    bartlett_window = torch.bartlett_window
    
    # 返回张量的方差和均值
    var_mean = torch.var_mean
    
    # 解线性最小二乘问题
    lstsq = torch.lstsq
    
    # 计算逻辑与
    logical_and = torch.logical_and
    
    # 计算多元正态分布的对数分布函数
    mvlgamma = torch.mvlgamma
    
    # 返回布莱克曼窗口的张量表示
    blackman_window = torch.blackman_window
    
    # 计算按位取反
    bitwise_not = torch.bitwise_not
    
    # 对方阵进行 Cholesky 逆
    cholesky_inverse = torch.
    ("block_diag", 136),
    # 创建一个元组，包含字符串 "block_diag" 和整数 136
    ("pca_lowrank", 124),
    # 创建一个元组，包含字符串 "pca_lowrank" 和整数 124
    ("absolute", 122),
    # 创建一个元组，包含字符串 "absolute" 和整数 122
    ("svd_lowrank", 108),
    # 创建一个元组，包含字符串 "svd_lowrank" 和整数 108
    ("neg", 2),
    # 创建一个元组，包含字符串 "neg" 和整数 2
# 定义一个包含神经网络功能名称和出现次数的列表
top_nn_functional = [
    ("nn.functional.softmax", 10522),  # softmax 函数出现次数
    ("nn.functional.relu", 8572),  # ReLU 函数出现次数
    ("nn.functional.interpolate", 7277),  # 插值函数出现次数
    ("nn.functional.pad", 5207),  # 填充函数出现次数
    ("nn.functional.log_softmax", 4699),  # 对数softmax 函数出现次数
    ("nn.functional.normalize", 2338),  # 归一化函数出现次数
    ("nn.functional.cross_entropy", 2083),  # 交叉熵函数出现次数
    ("nn.functional.grid_sample", 1970),  # 网格采样函数出现次数
    ("nn.functional.one_hot", 1967),  # one-hot 编码函数出现次数
    ("nn.functional.mse_loss", 1920),  # 均方误差损失函数出现次数
    ("nn.functional.conv2d", 1593),  # 二维卷积函数出现次数
    ("nn.functional.dropout", 1516),  # 随机失活函数出现次数
    ("nn.functional.softplus", 1385),  # Softplus 函数出现次数
    ("nn.functional.sigmoid", 1128),  # Sigmoid 函数出现次数
    ("nn.functional.linear", 1036),  # 线性函数出现次数
    ("nn.functional.gelu", 930),  # GELU 函数出现次数
    ("nn.functional.avg_pool2d", 899),  # 二维平均池化函数出现次数
    ("nn.functional.max_pool2d", 876),  # 二维最大池化函数出现次数
    ("nn.functional.nll_loss", 863),  # 负对数似然损失函数出现次数
    ("nn.functional.embedding", 737),  # 嵌入函数出现次数
    ("nn.functional.tanh", 664),  # 双曲正切函数出现次数
    ("nn.functional.leaky_relu", 640),  # 泄漏型ReLU函数出现次数
    ("nn.functional.adaptive_avg_pool2d", 633),  # 自适应二维平均池化函数出现次数
    ("nn.functional.cosine_similarity", 627),  # 余弦相似度函数出现次数
    ("nn.functional.unfold", 609),  # 展开函数出现次数
    ("nn.functional.conv1d", 596),  # 一维卷积函数出现次数
    ("nn.functional.binary_cross_entropy_with_logits", 591),  # 带Logits的二元交叉熵函数出现次数
    ("nn.functional.l1_loss", 571),  # L1 损失函数出现次数
    ("nn.functional.binary_cross_entropy", 492),  # 二元交叉熵函数出现次数
    ("nn.functional.elu", 416),  # ELU 函数出现次数
    ("nn.functional.batch_norm", 413),  # 批标准化函数出现次数
    ("nn.functional.upsample", 413),  # 上采样函数出现次数
    ("nn.functional.fold", 305),  # 折叠函数出现次数
    ("nn.functional.affine_grid", 298),  # 仿射网格函数出现次数
    ("nn.functional.max_pool1d", 297),  # 一维最大池化函数出现次数
    ("nn.functional.torch", 294),  # PyTorch 函数出现次数
    ("nn.functional.threshold", 263),  # 阈值函数出现次数
    ("nn.functional.smooth_l1_loss", 262),  # 平滑L1损失函数出现次数
    ("nn.functional.pairwise_distance", 253),  # 成对距离函数出现次数
    ("nn.functional.logsigmoid", 243),  # 对数Sigmoid函数出现次数
    ("nn.functional.adaptive_max_pool2d", 235),  # 自适应二维最大池化函数出现次数
    ("nn.functional.relu6", 213),  # ReLU6 函数出现次数
    ("nn.functional.pixel_shuffle", 209),  # 像素重排函数出现次数
    ("nn.functional.avg_pool3d", 203),  # 三维平均池化函数出现次数
    ("nn.functional.bilinear", 203),  # 双线性插值函数出现次数
    ("nn.functional.conv_transpose2d", 201),  # 二维转置卷积函数出现次数
    ("nn.functional.gumbel_softmax", 197),  # Gumbel-Softmax 函数出现次数
    ("nn.functional.max_unpool2d", 196),  # 二维最大反池化函数出现次数
    ("nn.functional.kl_div", 191),  # KL散度函数出现次数
    ("nn.functional.hardtanh", 189),  # 硬切线函数出现次数
    ("nn.functional.ctc_loss", 185),  # 连接时间分类损失函数出现次数
    ("nn.functional.layer_norm", 178),  # 层标准化函数出现次数
    ("nn.functional.conv3d", 172),  # 三维卷积函数出现次数
    ("nn.functional.max_unpool3d", 167),  # 三维最大反池化函数出现次数
    ("nn.functional.hardshrink", 165),  # 硬缩放函数出现次数
    ("nn.functional.hardswish", 156),  # 硬Swish 函数出现次数
    ("nn.functional.selu", 156),  # SELU 函数出现次数
    ("nn.functional.glu", 155),  # GLU 函数出现次数
    ("nn.functional.assert_int_or_pair", 150),  # 断言整数或成对函数出现次数
    ("nn.functional.hardsigmoid", 146),  # 硬Sigmoid 函数出现次数
    ("nn.functional.upsample_bilinear", 146),  # 双线性上采样函数出现次数
    ("nn.functional.max_pool3d", 140),  # 三维最大池化函数出现次数
    ("nn.functional.adaptive_avg_pool3d", 139),  # 自适应三维平均池化函数出现次数
    ("nn.functional.instance_norm", 124),  # 实例标准化函数出现次数
    ("nn.functional.embedding_bag", 122),  # 嵌入包函数出现次数
    ("nn.functional.upsample_nearest", 110),  # 最近邻上采样函数出现次数
    ("nn.functional.avg_pool1d", 105),  # 一维平均池化函数出现次数
    ("nn.functional.prelu", 102),  # PReLU 函数出现次数
    ("nn.functional.celu", 92),  # CELU 函数出现次数
    ("nn.functional.dropout2d", 86),  # 二维随机失活函数出现次数
    ("nn.functional.hinge_embedding_loss", 82),  # 铰链嵌入损失函数出现次数
    ("nn.functional.softsign", 81),  # Softsign 函数出现次数
    ("nn.functional.max_unpool1d", 74),  # 一维最大反池化函数出现次数
    ("nn.functional.silu", 74),  # SiLU 函数出现次数
    ("nn.functional.softshrink", 70),  # 软缩放函数出现次数
]
    # 导入 nn 模块中的函数，并附加它们的出现次数
    ("nn.functional.leaky_relu_", 68),  # leaky_relu_ 函数出现了 68 次
    ("nn.functional.softmin", 67),      # softmin 函数出现了 67 次
    ("nn.functional.channel_shuffle", 66),  # channel_shuffle 函数出现了 66 次
    ("nn.functional.multilabel_margin_loss", 66),  # multilabel_margin_loss 函数出现了 66 次
    ("nn.functional.dropout3d", 65),    # dropout3d 函数出现了 65 次
    ("nn.functional.multi_margin_loss", 65),  # multi_margin_loss 函数出现了 65 次
    ("nn.functional.lp_pool2d", 64),    # lp_pool2d 函数出现了 64 次
    ("nn.functional.conv_transpose1d", 62),  # conv_transpose1d 函数出现了 62 次
    ("nn.functional.triplet_margin_loss", 62),  # triplet_margin_loss 函数出现了 62 次
    ("nn.functional.tanhshrink", 61),   # tanhshrink 函数出现了 61 次
    ("nn.functional.adaptive_max_pool1d", 59),  # adaptive_max_pool1d 函数出现了 59 次
    ("nn.functional.cosine_embedding_loss", 58),  # cosine_embedding_loss 函数出现了 58 次
    ("nn.functional.multi_head_attention_forward", 58),  # multi_head_attention_forward 函数出现了 58 次
    ("nn.functional.max_pool1d_with_indices", 53),  # max_pool1d_with_indices 函数出现了 53 次
    ("nn.functional.poisson_nll_loss", 53),  # poisson_nll_loss 函数出现了 53 次
    ("nn.functional.margin_ranking_loss", 52),  # margin_ranking_loss 函数出现了 52 次
    ("nn.functional.soft_margin_loss", 52),  # soft_margin_loss 函数出现了 52 次
    ("nn.functional.adaptive_max_pool3d", 51),  # adaptive_max_pool3d 函数出现了 51 次
    ("nn.functional.group_norm", 51),    # group_norm 函数出现了 51 次
    ("nn.functional.local_response_norm", 51),  # local_response_norm 函数出现了 51 次
    ("nn.functional.multilabel_soft_margin_loss", 51),  # multilabel_soft_margin_loss 函数出现了 51 次
    ("nn.functional.relu_", 50),        # relu_ 函数出现了 50 次
    ("nn.functional.alpha_dropout", 49),  # alpha_dropout 函数出现了 49 次
    ("nn.functional.feature_alpha_dropout", 49),  # feature_alpha_dropout 函数出现了 49 次
    ("nn.functional.lp_pool1d", 49),    # lp_pool1d 函数出现了 49 次
    ("nn.functional.adaptive_max_pool1d_with_indices", 48),  # adaptive_max_pool1d_with_indices 函数出现了 48 次
    ("nn.functional.adaptive_max_pool2d_with_indices", 48),  # adaptive_max_pool2d_with_indices 函数出现了 48 次
    ("nn.functional.adaptive_max_pool3d_with_indices", 48),  # adaptive_max_pool3d_with_indices 函数出现了 48 次
    ("nn.functional.fractional_max_pool2d", 48),  # fractional_max_pool2d 函数出现了 48 次
    ("nn.functional.fractional_max_pool2d_with_indices", 48),  # fractional_max_pool2d_with_indices 函数出现了 48 次
    ("nn.functional.fractional_max_pool3d", 48),  # fractional_max_pool3d 函数出现了 48 次
    ("nn.functional.fractional_max_pool3d_with_indices", 48),  # fractional_max_pool3d_with_indices 函数出现了 48 次
    ("nn.functional.max_pool2d_with_indices", 48),  # max_pool2d_with_indices 函数出现了 48 次
    ("nn.functional.max_pool3d_with_indices", 48),  # max_pool3d_with_indices 函数出现了 48 次
    ("nn.functional.handle_torch_function", 47),  # handle_torch_function 函数出现了 47 次
    ("nn.functional.has_torch_function", 47),  # has_torch_function 函数出现了 47 次
    ("nn.functional.adaptive_avg_pool1d", 43),  # adaptive_avg_pool1d 函数出现了 43 次
    ("nn.functional.pdist", 43),        # pdist 函数出现了 43 次
    ("nn.functional.rrelu_", 37),       # rrelu_ 函数出现了 37 次
    ("nn.functional.elu_", 34),         # elu_ 函数出现了 34 次
    ("nn.functional.boolean_dispatch", 33),  # boolean_dispatch 函数出现了 33 次
    ("nn.functional.hardtanh_", 26),    # hardtanh_ 函数出现了 26 次
    ("nn.functional.triplet_margin_with_distance_loss", 23),  # triplet_margin_with_distance_loss 函数出现了 23 次
    ("nn.functional.selu_", 20),        # selu_ 函数出现了 20 次
    ("nn.functional.pixel_unshuffle", 19),  # pixel_unshuffle 函数出现了 19 次
    ("nn.functional.conv_transpose3d", 18),  # conv_transpose3d 函数出现了 18 次
    ("nn.functional.gaussian_nll_loss", 15),  # gaussian_nll_loss 函数出现了 15 次
    ("nn.functional.has_torch_function_unary", 15),  # has_torch_function_unary 函数出现了 15 次
    ("nn.functional.has_torch_function_variadic", 15),  # has_torch_function_variadic 函数出现了 15 次
    ("nn.functional.celu_", 13),        # celu_ 函数出现了 13 次
    ("nn.functional.huber_loss", 7),    # huber_loss 函数出现了 7 次
    ("nn.functional.mish", 4),          # mish 函数出现了 4 次
    ("nn.functional.threshold_", 3),    # threshold_ 函数出现了 3 次
    ("nn.functional.grad", 2),          # grad 函数出现了 2 次
    ("nn.functional.conv_tbc", 1),      # conv_tbc 函数出现了 1 次
    ("nn.functional.math", 1),          # math 函数出现了 1 次
# 定义一个列表，包含了神经网络模块的名称、代码行数和可能对应的功能模块
top_nn_module = [
    ("nn.Module", 927129, None),  # 神经网络模块的基类，无具体的功能模块对应
    ("nn.Linear", 530688, "nn.functional.linear"),  # 线性层，对应于线性函数
    ("nn.Sequential", 384968, None),  # 顺序容器，用于按顺序组织模块
    ("nn.Conv2d", 383320, "nn.functional.conv2d"),  # 二维卷积层，对应于二维卷积函数
    ("nn.ReLU", 318877, "nn.functional.relu"),  # ReLU 激活函数
    ("nn.BatchNorm2d", 233265, "nn.functional.batch_norm"),  # 二维批标准化层，对应于二维批标准化函数
    ("nn.Dropout", 179268, "nn.functional.dropout"),  # 随机置零的神经元（Dropout）层，对应于随机置零的函数
    ("nn.ModuleList", 171225, None),  # 模块列表容器，用于按顺序组织模块
    ("nn.Parameter", 153291, None),  # 参数张量
    ("nn.CrossEntropyLoss", 152696, "nn.functional.cross_entropy"),  # 交叉熵损失函数
    ("nn.MaxPool2d", 138619, "nn.functional.max_pool2d"),  # 二维最大池化层，对应于二维最大池化函数
    ("nn.Embedding", 111844, "nn.functional.embedding"),  # 嵌入层，对应于嵌入函数
    ("nn.DataParallel", 104238, None),  # 数据并行容器，用于在多个 GPU 上并行运行模块
    ("nn.MSELoss", 82954, "nn.functional.mse_loss"),  # 均方误差损失函数
    ("nn.Sigmoid", 75810, "nn.functional.sigmoid"),  # Sigmoid 激活函数
    ("nn.LeakyReLU", 65632, "nn.functional.leaky_relu"),  # LeakyReLU 激活函数
    ("nn.BatchNorm1d", 65374, "nn.functional.batch_norm"),  # 一维批标准化层，对应于一维批标准化函数
    ("nn.Softmax", 65114, "nn.functional.softmax"),  # Softmax 激活函数
    ("nn.Tanh", 59445, "nn.functional.tanh"),  # Tanh 激活函数
    ("nn.AdaptiveAvgPool2d", 59071, "nn.functional.adaptive_avg_pool2d"),  # 自适应平均池化层，对应于自适应平均池化函数
    ("nn.AvgPool2d", 58377, "nn.functional.avg_pool2d"),  # 二维平均池化层，对应于二维平均池化函数
    ("nn.ConvTranspose2d", 57524, "nn.functional.conv_transpose2d"),  # 二维转置卷积层，对应于二维转置卷积函数
    ("nn.LSTM", 57411, None),  # 长短时记忆网络（LSTM）层
    ("nn.Conv1d", 41108, "nn.functional.conv1d"),  # 一维卷积层，对应于一维卷积函数
    ("nn.LayerNorm", 36089, "nn.functional.layer_norm"),  # 层归一化层，对应于层归一化函数
    ("nn.BCELoss", 34005, "nn.functional.binary_cross_entropy"),  # 二元交叉熵损失函数
    ("nn.Upsample", 32527, "nn.functional.interpolate"),  # 上采样层，对应于插值函数
    ("nn.BCEWithLogitsLoss", 29944, "nn.functional.binary_cross_entropy_with_logits"),  # 带 logits 的二元交叉熵损失函数
    ("nn.GRU", 25421, None),  # 门控循环单元（GRU）层
    ("nn.Dropout2d", 23512, "nn.functional.dropout2d"),  # 二维随机置零的神经元（Dropout）层，对应于二维随机置零的函数
    ("nn.LogSoftmax", 22897, "nn.functional.log_softmax"),  # 对数 Softmax 操作
    ("nn.L1Loss", 22778, "nn.functional.l1_loss"),  # L1 损失函数
    ("nn.GroupNorm", 22183, "nn.functional.group_norm"),  # 分组归一化层，对应于分组归一化函数
    ("nn.NLLLoss", 21751, "nn.functional.nll_loss"),  # 负对数似然损失函数
    ("nn.Conv3d", 20874, "nn.functional.conv3d"),  # 三维卷积层，对应于三维卷积函数
    ("nn.Identity", 17911, None),  # 恒等映射层，不进行任何操作
    ("nn.InstanceNorm2d", 16426, "nn.functional.instance_norm"),  # 实例标准化层，对应于实例标准化函数
    ("nn.BatchNorm3d", 16378, "nn.functional.batch_norm"),  # 三维批标准化层，对应于三维批标准化函数
    ("nn.PReLU", 13472, "nn.functional.prelu"),  # 参数化的 ReLU 激活函数
    ("nn.ReLU6", 12622, "nn.functional.relu6"),  # ReLU6 激活函数
    ("nn.ELU", 12508, "nn.functional.elu"),  # ELU 激活函数
    ("nn.LSTMCell", 10885, None),  # 单个长短时记忆网络（LSTM）单元
    ("nn.Flatten", 10384, "torch.flatten"),  # 展平操作，对应于展平函数
    ("nn.ModuleDict", 10255, None),  # 模块字典容器，用于按键值对组织模块
    ("nn.ReflectionPad2d", 9954, "nn.functional.pad"),  # 二维反射填充层，对应于二维填充函数
    ("nn.MaxPool3d", 9526, "nn.functional.max_pool3d"),  # 三维最大池化层，对应于三维最大池化函数
    ("nn.MaxPool1d", 9154, "nn.functional.max_pool1d"),  # 一维最大池化层，对应于一维最大池化函数
    ("nn.RNN", 9154, None),  # 循环神经网络（RNN）层
    ("nn.ZeroPad2d", 8847, "nn.functional.pad"),  # 二维零填充层，对应于二维填充函数
    ("nn.ParameterList", 7702, None),  # 参数列表容器，用于按顺序组织参数
    ("nn.SyncBatchNorm", 6814, None),  # 同步批标准化层，用于在多 GPU 上同步运行模块
    ("nn.PixelShuffle", 6571, "nn.functional.pixel_shuffle"),  # 像素重排操作，对应于像素重排函数
    ("nn.SmoothL1Loss", 6517, "nn.functional.smooth_l1_loss"),  # 平滑 L1 损失函数
    ("nn.Hardswish", 6458, "nn.functional.hardswish"),  # Hardswish 激活函数
    ("nn.AdaptiveMaxPool2d", 6071, "nn.functional.adaptive_max_pool2d"),  # 自适应最大池化层，对应于自适应最大池化函数
    ("nn.SEL
    # 导入 nn 模块中的 ReplicationPad2d 类，并且指定其在文档中出现的次数为 5600 次，同时指定其功能替代为 nn.functional.pad 函数
    ("nn.ReplicationPad2d", 5600, "nn.functional.pad"),
    # 导入 nn 模块中的 KLDivLoss 类，并且指定其在文档中出现的次数为 5541 次，同时指定其功能替代为 nn.functional.kl_div 函数
    ("nn.KLDivLoss", 5541, "nn.functional.kl_div"),
    # 导入 nn 模块中的 ConvTranspose1d 类，并且指定其在文档中出现的次数为 5183 次，同时指定其功能替代为 nn.functional.conv_transpose1d 函数
    ("nn.ConvTranspose1d", 5183, "nn.functional.conv_transpose1d"),
    # 导入 nn 模块中的 Softplus 类，并且指定其在文档中出现的次数为 5120 次，同时指定其功能替代为 nn.functional.softplus 函数
    ("nn.Softplus", 5120, "nn.functional.softplus"),
    # 导入 nn 模块中的 SiLU 类，并且指定其在文档中出现的次数为 4895 次，同时指定其功能替代为 nn.functional.silu 函数
    ("nn.SiLU", 4895, "nn.functional.silu"),
    # 导入 nn 模块中的 AvgPool3d 类，并且指定其在文档中出现的次数为 4523 次，同时指定其功能替代为 nn.functional.avg_pool3d 函数
    ("nn.AvgPool3d", 4523, "nn.functional.avg_pool3d"),
    # 导入 nn 模块中的 CosineSimilarity 类，并且指定其在文档中出现的次数为 4058 次，同时指定其功能替代为 nn.functional.cosine_similarity 函数
    ("nn.CosineSimilarity", 4058, "nn.functional.cosine_similarity"),
    # 导入 nn 模块中的 GELU 类，并且指定其在文档中出现的次数为 3932 次，同时指定其功能替代为 nn.functional.gelu 函数
    ("nn.GELU", 3932, "nn.functional.gelu"),
    # 导入 nn 模块中的 UpsamplingBilinear2d 类，并且指定其在文档中出现的次数为 3673 次，同时指定其功能替代为 nn.functional.interpolate 函数
    ("nn.UpsamplingBilinear2d", 3673, "nn.functional.interpolate"),
    # 导入 nn 模块中的 InstanceNorm1d 类，并且指定其在文档中出现的次数为 3658 次，同时指定其功能替代为 nn.functional.instance_norm 函数
    ("nn.InstanceNorm1d", 3658, "nn.functional.instance_norm"),
    # 导入 nn 模块中的 Transformer 类，并且指定其在文档中出现的次数为 3604 次，同时没有指定功能替代（None）
    ("nn.Transformer", 3604, None),
    # 导入 nn 模块中的 MultiheadAttention 类，并且指定其在文档中出现的次数为 3435 次，同时指定其功能替代为 nn.functional.multi_head_attention_forward 函数
    ("nn.MultiheadAttention", 3435, "nn.functional.multi_head_attention_forward"),
    # 导入 nn 模块中的 AvgPool1d 类，并且指定其在文档中出现的次数为 3195 次，同时指定其功能替代为 nn.functional.avg_pool1d 函数
    ("nn.AvgPool1d", 3195, "nn.functional.avg_pool1d"),
    # 导入 nn 模块中的 Dropout3d 类，并且指定其在文档中出现的次数为 2964 次，同时指定其功能替代为 nn.functional.dropout3d 函数
    ("nn.Dropout3d", 2964, "nn.functional.dropout3d"),
    # 导入 nn 模块中的 AdaptiveAvgPool3d 类，并且指定其在文档中出现的次数为 2915 次，同时指定其功能替代为 nn.functional.adaptive_avg_pool3d 函数
    ("nn.AdaptiveAvgPool3d", 2915, "nn.functional.adaptive_avg_pool3d"),
    # 导入 nn 模块中的 InstanceNorm3d 类，并且指定其在文档中出现的次数为 2893 次，同时指定其功能替代为 nn.functional.instance_norm 函数
    ("nn.InstanceNorm3d", 2893, "nn.functional.instance_norm"),
    # 导入 nn 模块中的 Hardtanh 类，并且指定其在文档中出现的次数为 2613 次，同时指定其功能替代为 nn.functional.hardtanh 函数
    ("nn.Hardtanh", 2613, "nn.functional.hardtanh"),
    # 导入 nn 模块中的 MarginRankingLoss 类，并且指定其在文档中出现的次数为 2568 次，同时指定其功能替代为 nn.functional.margin_ranking_loss 函数
    ("nn.MarginRankingLoss", 2568, "nn.functional.margin_ranking_loss"),
    # 导入 nn 模块中的 GLU 类，并且指定其在文档中出现的次数为 2526 次，同时指定其功能替代为 nn.functional.glu 函数
    ("nn.GLU", 2526, "nn.functional.glu"),
    # 导入 nn 模块中的 AdaptiveAvgPool1d 类，并且指定其在文档中出现的次数为 2481 次，同时指定其功能替代为 nn.functional.adaptive_avg_pool1d 函数
    ("nn.AdaptiveAvgPool1d", 2481, "nn.functional.adaptive_avg_pool1d"),
    # 导入 nn 模块中的 EmbeddingBag 类，并且指定其在文档中出现的次数为 2344 次，同时指定其功能替代为 nn.functional.embedding_bag 函数
    ("nn.EmbeddingBag", 2344, "nn.functional.embedding_bag"),
    # 导入 nn 模块中的 TransformerEncoderLayer 类，并且指定其在文档中出现的次数为 2292 次，同时没有指定功能替代（None）
    ("nn.TransformerEncoderLayer", 2292, None),
    # 导入 nn 模块中的 TransformerEncoder 类，并且指定其在文档中出现的次数为 2091 次，同时没有指定功能替代（None）
    ("nn.TransformerEncoder", 2091, None),
    # 导入 nn 模块中的 MaxUnpool2d 类，并且指定其在文档中出现的次数为 2031 次，同时指定其功能替代为 nn.functional.max_unpool2d 函数
    ("nn.MaxUnpool2d", 2031, "nn.functional.max_unpool2d"),
    # 导入 nn 模块中的 UpsamplingNearest2d 类，并且指定其在文档中出现的次数为 2004 次，同时指定其功能替代为 nn.functional.interpolate 函数
    ("nn.UpsamplingNearest2d", 2004, "nn.functional.interpolate"),
    # 导入 nn 模块中的 ConstantPad1d 类，并且指定其在文档中出现的次数为 1904 次，同时指定其功能替代为 nn.functional.pad 函数
    ("nn.ConstantPad1d", 1904, "nn.functional.pad"),
    # 导入 nn 模块中的 ConstantPad2d 类，并且指定其在文档中出现的次数为 1791 次，同时指定其功能替代为 nn.functional.pad 函数
    ("nn.ConstantPad2d", 1791, "nn.functional.pad"),
    # 导入
    # 创建一个包含元组的列表，每个元组包含模块名、数量和对应的函数或 None
    ("nn.AlphaDropout", 710, "nn.functional.alpha_dropout"),
    ("nn.Tanhshrink", 681, "nn.functional.tanhshrink"),
    ("nn.PoissonNLLLoss", 676, "nn.functional.poisson_nll_loss"),
    ("nn.MaxUnpool3d", 660, "nn.functional.max_unpool3d"),
    ("nn.Fold", 630, "nn.functional.fold"),
    ("nn.MultiMarginLoss", 622, "nn.functional.multi_margin_loss"),
    ("nn.TransformerDecoderLayer", 614, None),
    ("nn.TransformerDecoder", 607, None),
    ("nn.Hardshrink", 592, "nn.functional.hardshrink"),
    ("nn.ConstantPad3d", 582, "nn.functional.pad"),
    ("nn.MultiLabelMarginLoss", 580, "nn.functional.multilabel_margin_loss"),
    ("nn.LPPool2d", 550, "nn.functional.lp_pool2d"),
    ("nn.Softmin", 537, "nn.functional.softmin"),
    ("nn.MaxUnpool1d", 518, "nn.functional.max_unpool1d"),
    ("nn.FractionalMaxPool2d", 484, "nn.functional.fractional_max_pool2d"),
    ("nn.Hardsigmoid", 477, "nn.functional.hardsigmoid"),
    ("nn.ReplicationPad3d", 470, "nn.functional.pad"),
    ("nn.HingeEmbeddingLoss", 442, "nn.functional.hinge_embedding_loss"),
    ("nn.LPPool1d", 386, "nn.functional.lp_pool1d"),
    ("nn.FractionalMaxPool3d", 252, "nn.functional.fractional_max_pool3d"),
    ("nn.Container", 217, None),
    ("nn.Unflatten", 206, "nn.functional.unflatten"),
    ("nn.FeatureAlphaDropout", 136, "nn.functional.feature_alpha_dropout"),
    (
        "nn.TripletMarginWithDistanceLoss",
        107,
        "nn.functional.triplet_margin_with_distance_loss",
    ),
    ("nn.ChannelShuffle", 90, "nn.functional.channel_shuffle"),
    ("nn.RNNCellBase", 88, None),
    ("nn.LazyLinear", 81, "nn.functional.linear"),
    ("nn.UninitializedParameter", 60, None),
    ("nn.CrossMapLRN2d", 59, None),
    ("nn.GaussianNLLLoss", 55, "nn.functional.gaussian_nll_loss"),
    ("nn.PixelUnshuffle", 45, "nn.functional.pixel_unshuffle"),
    ("nn.Mish", 31, "nn.functional.mish"),
    ("nn.ReflectionPad3d", 22, "nn.functional.pad"),
    ("nn.HuberLoss", 18, "nn.functional.huber_loss"),
    ("nn.LazyConv2d", 15, None),
    ("nn.LazyConv1d", 9, None),
    ("nn.LazyConv3d", 8, None),
    ("nn.LazyConvTranspose1d", 8, None),
    ("nn.LazyConvTranspose2d", 8, None),
    ("nn.LazyConvTranspose3d", 8, None),
    ("nn.LazyBatchNorm1d", 3, None),
    ("nn.LazyBatchNorm2d", 3, None),
    ("nn.LazyBatchNorm3d", 3, None),
    ("nn.UninitializedBuffer", 3, None),
# 仅包含方法操作的列表，这些操作比较难以获取排名
method_only_ops = [
    "bfloat16",        # bfloat16 数据类型
    "bool",            # 布尔类型
    "byte",            # 字节类型
    "char",            # 字符类型
    "contiguous",      # 返回一个连续的张量
    "cpu",             # 将张量转移到 CPU 上
    "cuda",            # 将张量转移到 CUDA 上
    "detach",          # 返回一个新的张量，从当前计算图中分离出来
    "double",          # 双精度浮点数类型
    "expand",          # 扩展张量的大小
    "expand_as",       # 将张量扩展为指定大小
    "float",           # 单精度浮点数类型
    "get_device",      # 获取张量所在的设备索引
    "half",            # 半精度浮点数类型
    "hardshrink",      # 执行硬收缩函数操作
    "index_add",       # 执行索引加操作
    "index_copy",      # 执行索引复制操作
    "index_fill",      # 执行索引填充操作
    "index_put",       # 执行索引赋值操作
    "int",             # 整数类型
    "is_contiguous",   # 判断张量是否是连续的
    "is_pinned",       # 判断张量是否被固定在内存中
    "is_set_to",       # 判断张量是否被设置为指定值
    "is_shared",       # 判断张量是否是共享的
    "is_signed",       # 判断张量是否有符号
    "item",            # 获取张量的标量值
    "long",            # 长整型类型
    "masked_scatter",  # 执行掩码散射操作
    "masked_fill",     # 执行掩码填充操作
    "narrow_copy",     # 执行窄复制操作
    "numpy",           # 将张量转换为 NumPy 数组
    "pin_memory",      # 将张量固定在内存中
    "repeat",          # 执行张量重复操作
    "reshape_as",      # 将张量重塑为指定张量的形状
    "select",          # 执行选择操作
    "short",           # 短整型类型
    "storage_offset",  # 获取张量数据的偏移量
    "sum_to_size",     # 按照指定尺寸求和
    "to",              # 张量的类型转换
    "to_mkldnn",       # 将张量转换为 MKL-DNN 张量
    "tolist",          # 将张量转换为 Python 列表
    "type",            # 获取张量的数据类型
    "type_as",         # 将张量转换为指定类型的张量
    "unfold",          # 执行展开操作
    "view",            # 执行视图操作
    "view_as",         # 将张量视图为指定张量的形状
]

def get_nn_functional_top_list():
    top_nn_functional_ = dict(top_nn_functional)
    for _, count, functional_name in top_nn_module:
        if functional_name is None:
            continue
        if functional_name == "torch.flatten":
            continue
        if functional_name not in top_nn_functional_:
            top_nn_functional_[functional_name] = count
        else:
            top_nn_functional_[functional_name] += count

    top_nn_functional_ = list(top_nn_functional_.items())
    top_nn_functional_.sort(key=operator.itemgetter(1), reverse=True)
    return top_nn_functional_

# 用于存储每个操作的使用次数的字典
usage_count = {}

# 遍历并统计获取到的神经网络功能列表中的每个项的使用次数
for k, v in get_nn_functional_top_list():
    usage_count[k] = v

# 遍历并将顶级 Torch 操作的使用次数添加到 usage_count 字典中
for k, v in top_torch:
    usage_count[k] = v
```