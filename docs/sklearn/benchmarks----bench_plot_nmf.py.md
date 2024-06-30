# `D:\src\scipysrc\scikit-learn\benchmarks\bench_plot_nmf.py`

```
    beta : float
        Factor by which the step size will be multiplied during backtracking
        line search. Should be in the interval (0, 1).
    Returns
    -------
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        Number of iterations run by the algorithm.
    """
    n_samples, n_features = X.shape

    # Initialize variables
    WtX = safe_sparse_dot(W.T, X)
    WtW = safe_sparse_dot(W.T, W)

    # To avoid the premature convergence, compute epsilon
    # that will be used instead of 0 where it's enough
    # small.
    eps = np.finfo(float).eps

    # Projected gradient descent algorithm
    for n_iter in range(1, max_iter + 1):
        # The matrix `H` is to be updated
        # according to the `W` matrix
        grad_W, grad_H = _beta_divergence(X, W, H, beta)
        if _norm(grad_H) < eps:
            break
        WtX = safe_sparse_dot(W.T, X)
        WtW = safe_sparse_dot(W.T, W)

        # A `threshold` is used to limit the maximum iteration count
        # of the algorithm to the sum of `tol` values.
        ? To proceed
    # WtX 是 W 的转置与 X 的矩阵乘积
    WtX = safe_sparse_dot(W.T, X)
    # WtW 是 W 的转置与 W 的矩阵乘积
    WtW = np.dot(W.T, W)

    # 在论文中论证的值（alpha 被重命名为 gamma）
    gamma = 1
    # 迭代更新 H 的非负最小二乘问题的解
    for n_iter in range(1, max_iter + 1):
        # 计算梯度 grad
        grad = np.dot(WtW, H) - WtX
        # 根据 alpha 和 l1_ratio 条件调整梯度 grad
        if alpha > 0 and l1_ratio == 1.0:
            grad += alpha
        elif alpha > 0:
            grad += alpha * (l1_ratio + (1 - l1_ratio) * H)

        # 通过与布尔数组相乘加速计算，比直接索引 grad 更快
        if _norm(grad * np.logical_or(grad < 0, H > 0)) < tol:
            break

        # 记录当前的 H 值
        Hp = H

        # 内部迭代求解循环，最多 20 次
        for inner_iter in range(20):
            # 梯度步长
            Hn = H - gamma * grad
            # 投影步骤，保证 Hn 元素非负
            Hn *= Hn > 0
            # 计算 Hn 和 H 之间的差值 d
            d = Hn - H
            # 计算 grad 和 d 的内积 gradd
            gradd = np.dot(grad.ravel(), d.ravel())
            # 计算 dQd，其中 Q 是 WtW 与 d 的矩阵乘积
            dQd = np.dot(np.dot(WtW, d).ravel(), d.ravel())
            # 检查是否满足足够减少条件
            suff_decr = (1 - sigma) * gradd + 0.5 * dQd < 0

            # 第一次迭代时记录是否需要减少 gamma
            if inner_iter == 0:
                decr_gamma = not suff_decr

            # 根据 suff_decr 和 decr_gamma 更新 gamma 和 H
            if decr_gamma:
                if suff_decr:
                    H = Hn
                    break
                else:
                    gamma *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                gamma /= beta
                Hp = Hn

    # 如果达到最大迭代次数则发出警告
    if n_iter == max_iter:
        warnings.warn("Iteration limit reached in nls subproblem.", ConvergenceWarning)

    # 返回最终解 H，梯度 grad，迭代次数 n_iter
    return H, grad, n_iter
def _fit_projected_gradient(X, W, H, tol, max_iter, nls_max_iter, alpha, l1_ratio):
    # 计算W的梯度：W的梯度是W乘以H乘以H的转置，减去X乘以H的转置
    gradW = np.dot(W, np.dot(H, H.T)) - safe_sparse_dot(X, H.T, dense_output=True)
    # 计算H的梯度：H的梯度是W的转置乘以W乘以H，减去W的转置乘以X
    gradH = np.dot(np.dot(W.T, W), H) - safe_sparse_dot(W.T, X, dense_output=True)

    init_grad = squared_norm(gradW) + squared_norm(gradH.T)
    # max(0.001, tol)用来强制交替最小化W和H
    tolW = max(0.001, tol) * np.sqrt(init_grad)
    tolH = tolW

    for n_iter in range(1, max_iter + 1):
        # 停止条件如论文所述
        proj_grad_W = squared_norm(gradW * np.logical_or(gradW < 0, W > 0))
        proj_grad_H = squared_norm(gradH * np.logical_or(gradH < 0, H > 0))

        if (proj_grad_W + proj_grad_H) / init_grad < tol**2:
            break

        # 更新W
        Wt, gradWt, iterW = _nls_subproblem(
            X.T, H.T, W.T, tolW, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio
        )
        W, gradW = Wt.T, gradWt.T

        if iterW == 1:
            tolW = 0.1 * tolW

        # 更新H
        H, gradH, iterH = _nls_subproblem(
            X, W, H, tolH, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio
        )
        if iterH == 1:
            tolH = 0.1 * tolH

    H[H == 0] = 0  # 修正负零值

    if n_iter == max_iter:
        Wt, _, _ = _nls_subproblem(
            X.T, H.T, W.T, tolW, nls_max_iter, alpha=alpha, l1_ratio=l1_ratio
        )
        W = Wt.T

    return W, H, n_iter


class _PGNMF(NMF):
    """使用投影梯度求解器的非负矩阵分解（NMF）。

    此类是私有的，仅供比较目的使用。
    可能会在不通知的情况下更改或消失。

    """

    def __init__(
        self,
        n_components=None,
        solver="pg",
        init=None,
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha=0.0,
        l1_ratio=0.0,
        nls_max_iter=10,
    ):
        super().__init__(
            n_components=n_components,
            init=init,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha,
            alpha_H=alpha,
            l1_ratio=l1_ratio,
        )
        self.nls_max_iter = nls_max_iter

    def fit(self, X, y=None, **params):
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        check_is_fitted(self)
        H = self.components_
        W, _, self.n_iter_ = self._fit_transform(X, H=H, update_H=False)
        return W

    def inverse_transform(self, W):
        check_is_fitted(self)
        return np.dot(W, self.components_)

    def fit_transform(self, X, y=None, W=None, H=None):
        W, H, self.n_iter = self._fit_transform(X, W=W, H=H, update_H=True)
        self.components_ = H
        return W
    # 拟合或转换数据，执行NMF算法的核心函数，根据参数更新或初始化W和H矩阵
    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        # 检查并转换输入数据X，确保其为稀疏矩阵格式
        X = check_array(X, accept_sparse=("csr", "csc"))
        # 检查X中是否包含非负数值，用于NMF算法
        check_non_negative(X, "NMF (input X)")

        # 获取输入数据的样本数和特征数
        n_samples, n_features = X.shape
        # 确定要拟合的分量数，默认为特征数
        n_components = self.n_components
        if n_components is None:
            n_components = n_features

        # 检查分量数是否为正整数
        if not isinstance(n_components, numbers.Integral) or n_components <= 0:
            raise ValueError(
                "Number of components must be a positive integer; got (n_components=%r)"
                % n_components
            )
        # 检查最大迭代次数是否为正整数
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iterations must be a positive "
                "integer; got (max_iter=%r)" % self.max_iter
            )
        # 检查停止条件的容差是否为正数
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        # 检查是否提供了W和H，若未提供则进行初始化
        if self.init == "custom" and update_H:
            _check_init(H, (n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, n_components), "NMF (input W)")
        elif not update_H:
            _check_init(H, (n_components, n_features), "NMF (input H)")
            # 若不需要更新H矩阵，则初始化W矩阵为全零
            W = np.zeros((n_samples, n_components))
        else:
            # 否则，根据输入数据和初始化方法初始化W和H矩阵
            W, H = _initialize_nmf(
                X, n_components, init=self.init, random_state=self.random_state
            )

        # 若需要更新H矩阵，则执行拟合与转换操作
        if update_H:  # fit_transform
            # 使用投影梯度法拟合W和H矩阵
            W, H, n_iter = _fit_projected_gradient(
                X,
                W,
                H,
                self.tol,
                self.max_iter,
                self.nls_max_iter,
                self.alpha,
                self.l1_ratio,
            )
        else:  # transform
            # 若不需要更新H矩阵，则执行变换操作，求解最优的W矩阵
            Wt, _, n_iter = _nls_subproblem(
                X.T,
                H.T,
                W.T,
                self.tol,
                self.nls_max_iter,
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
            )
            W = Wt.T

        # 若迭代次数达到最大值且容差仍大于零，则发出警告信息
        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iteration %d reached. Increase it"
                " to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        # 返回计算得到的W、H矩阵以及迭代次数
        return W, H, n_iter
#################
# End of _PGNMF #
#################


def plot_results(results_df, plot_name):
    if results_df is None:
        return None

    # 创建一个宽度为 16 英寸，高度为 6 英寸的图形
    plt.figure(figsize=(16, 6))
    # 设定颜色和标记符号的序列
    colors = "bgr"
    markers = "ovs"
    # 创建一个子图，共有 1 行 3 列，ax 是第一个子图的句柄
    ax = plt.subplot(1, 3, 1)
    # 遍历数据框中 "init" 列的唯一值
    for i, init in enumerate(np.unique(results_df["init"])):
        # 创建一个子图，共有 1 行 3 列，共享 x 和 y 轴
        plt.subplot(1, 3, i + 1, sharex=ax, sharey=ax)
        # 遍历数据框中 "method" 列的唯一值
        for j, method in enumerate(np.unique(results_df["method"])):
            # 创建一个布尔掩码，选择符合条件的数据项
            mask = np.logical_and(
                results_df["init"] == init, results_df["method"] == method
            )
            selected_items = results_df[mask]

            # 绘制散点图，x 轴为时间，y 轴为损失值，使用不同的颜色和标记符号表示不同的方法
            plt.plot(
                selected_items["time"],
                selected_items["loss"],
                color=colors[j % len(colors)],
                ls="-",
                marker=markers[j % len(markers)],
                label=method,
            )

        # 添加图例，位于最佳位置，字体大小为 'x-small'
        plt.legend(loc=0, fontsize="x-small")
        plt.xlabel("Time (s)")  # 设置 x 轴标签
        plt.ylabel("loss")      # 设置 y 轴标签
        plt.title("%s" % init)  # 设置子图标题为 init 的值
    plt.suptitle(plot_name, fontsize=16)  # 设置总标题为 plot_name，字体大小为 16


@ignore_warnings(category=ConvergenceWarning)
# 使用 joblib 缓存结果。
# X_shape 在参数中指定，用于避免对 X 进行哈希处理
@mem.cache(ignore=["X", "W0", "H0"])
def bench_one(
    name, X, W0, H0, X_shape, clf_type, clf_params, init, n_components, random_state
):
    # 复制初始值 W0 和 H0
    W = W0.copy()
    H = H0.copy()

    # 使用指定的分类器类型和参数创建分类器对象 clf
    clf = clf_type(**clf_params)
    st = time()  # 记录开始时间
    # 使用分类器拟合数据 X，并同时更新 W 和 H
    W = clf.fit_transform(X, W=W, H=H)
    end = time()  # 记录结束时间
    H = clf.components_  # 获取更新后的成分 H

    # 计算使用 beta 散度（beta divergence）计算的损失值
    this_loss = _beta_divergence(X, W, H, 2.0, True)
    duration = end - st  # 计算运行时间
    return this_loss, duration  # 返回损失值和运行时间


def run_bench(X, clfs, plot_name, n_components, tol, alpha, l1_ratio):
    start = time()  # 记录开始时间
    results = []  # 初始化结果列表
    # 遍历分类器列表 clfs
    for name, clf_type, iter_range, clf_params in clfs:
        print("Training %s:" % name)
        # 遍历初始化方式列表
        for rs, init in enumerate(("nndsvd", "nndsvdar", "random")):
            print("    %s %s: " % (init, " " * (8 - len(init))), end="")
            # 使用 _initialize_nmf 函数初始化 NMF 的 W 和 H
            W, H = _initialize_nmf(X, n_components, init, 1e-6, rs)

            # 遍历迭代次数列表
            for max_iter in iter_range:
                # 更新分类器参数
                clf_params["alpha"] = alpha
                clf_params["l1_ratio"] = l1_ratio
                clf_params["max_iter"] = max_iter
                clf_params["tol"] = tol
                clf_params["random_state"] = rs
                clf_params["init"] = "custom"
                clf_params["n_components"] = n_components

                # 调用 bench_one 函数执行单次性能评估
                this_loss, duration = bench_one(
                    name, X, W, H, X.shape, clf_type, clf_params, init, n_components, rs
                )

                init_name = "init='%s'" % init
                results.append((name, this_loss, duration, init_name))
                # 打印损失值和时间信息
                print(".", end="")
                sys.stdout.flush()
            print(" ")

    # 使用 pandas 创建数据框，整理结果
    results_df = pandas.DataFrame(results, columns="method loss time init".split())
    # 打印总耗时，格式化输出并计算时间差
    print("Total time = %0.3f sec\n" % (time() - start))
    
    # 绘制结果图表
    plot_results(results_df, plot_name)
    # 返回结果数据框
    return results_df
# 定义函数 load_20news，用于加载并处理 20 newsgroups 数据集
def load_20news():
    # 打印加载过程中的提示信息
    print("Loading 20 newsgroups dataset")
    print("-----------------------------")
    # 导入 fetch_20newsgroups 函数从 sklearn.datasets 中
    from sklearn.datasets import fetch_20newsgroups

    # 使用 fetch_20newsgroups 函数加载数据集，同时设置参数进行处理
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
    )
    # 使用 TfidfVectorizer 对文本数据进行 TF-IDF 向量化处理
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
    # 对文本数据进行 TF-IDF 变换，得到稀疏矩阵 tfidf
    tfidf = vectorizer.fit_transform(dataset.data)
    # 返回处理后的稀疏矩阵 tfidf
    return tfidf


# 定义函数 load_faces，用于加载 Olivetti 人脸数据集
def load_faces():
    # 打印加载过程中的提示信息
    print("Loading Olivetti face dataset")
    print("-----------------------------")
    # 导入 fetch_olivetti_faces 函数从 sklearn.datasets 中
    from sklearn.datasets import fetch_olivetti_faces

    # 使用 fetch_olivetti_faces 函数加载数据集，同时设置参数进行处理
    faces = fetch_olivetti_faces(shuffle=True)
    # 返回人脸数据集的数据部分 faces.data
    return faces.data


# 定义函数 build_clfs，用于构建多个分类器配置
def build_clfs(cd_iters, pg_iters, mu_iters):
    # 构建包含多个分类器配置的列表 clfs
    clfs = [
        ("Coordinate Descent", NMF, cd_iters, {"solver": "cd"}),
        ("Projected Gradient", _PGNMF, pg_iters, {"solver": "pg"}),
        ("Multiplicative Update", NMF, mu_iters, {"solver": "mu"}),
    ]
    # 返回配置好的分类器列表 clfs
    return clfs


# 主程序入口
if __name__ == "__main__":
    # 初始化参数设置
    alpha = 0.0
    l1_ratio = 0.5
    n_components = 10
    tol = 1e-15

    # 第一个基准测试：使用 20 newsgroups 数据集进行测试
    plot_name = "20 Newsgroups sparse dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 6)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_20news = load_20news()
    # 调用 run_bench 函数进行性能测试，并传入相关参数
    run_bench(X_20news, clfs, plot_name, n_components, tol, alpha, l1_ratio)

    # 第二个基准测试：使用 Olivetti 人脸数据集进行测试
    plot_name = "Olivetti Faces dense dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 12)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_faces = load_faces()
    # 再次调用 run_bench 函数进行性能测试，并传入相关参数
    run_bench(
        X_faces,
        clfs,
        plot_name,
        n_components,
        tol,
        alpha,
        l1_ratio,
    )

    # 显示图形结果
    plt.show()
```