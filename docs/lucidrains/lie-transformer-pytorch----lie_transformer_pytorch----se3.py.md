# `.\lucidrains\lie-transformer-pytorch\lie_transformer_pytorch\se3.py`

```
# 从 math 模块中导入 pi 常数
from math import pi
# 导入 torch 模块
import torch
# 从 functools 模块中导入 wraps 装饰器
from functools import wraps
# 从 torch 模块中导入 acos, atan2, cos, sin 函数
from torch import acos, atan2, cos, sin
# 从 einops 模块中导入 rearrange, repeat 函数

# 常量
THRES = 7e-2

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回张量的设备和数据类型
def to(t):
    return {'device': t.device, 'dtype': t.dtype}

# Taylor 展开函数
def taylor(thres):
    def outer(fn):
        @wraps(fn)
        def inner(x):
            usetaylor = x.abs() < THRES
            taylor_expanded, full = fn(x, x * x)
            return torch.where(usetaylor, taylor_expanded, full)
        return inner
    return outer

# 用于解析指数映射的辅助函数。在 x=0 附近使用 Taylor 展开
# 参考 http://ethaneade.com/lie_groups.pdf 进行推导

# sinc 函数的 Taylor 展开
@taylor(THRES)
def sinc(x, x2):
    """ sin(x)/x """
    texpand = 1-x2/6*(1-x2/20*(1-x2/42))
    full = sin(x) / x
    return texpand, full

# sincc 函数的 Taylor 展开
@taylor(THRES)
def sincc(x, x2):
    """ (1-sinc(x))/x^2"""
    texpand = 1/6*(1-x2/20*(1-x2/42*(1-x2/72)))
    full = (x-sin(x)) / x**3
    return texpand, full

# cosc 函数的 Taylor 展开
@taylor(THRES)
def cosc(x, x2):
    """ (1-cos(x))/x^2"""
    texpand = 1/2*(1-x2/12*(1-x2/30*(1-x2/56)))
    full = (1-cos(x)) / x2
    return texpand, full

# coscc 函数的 Taylor 展开
@taylor(THRES)
def coscc(x, x2):
    texpand = 1/12*(1+x2/60*(1+x2/42*(1+x2/40)))
    costerm = (2*(1-cos(x))).clamp(min=1e-6)
    full = (1-x*sin(x)/costerm) / x2
    return texpand, full

# sinc_inv 函数的 Taylor 展开
@taylor(THRES)
def sinc_inv(x, _):
    texpand = 1+(1/6)*x**2 +(7/360)*x**4
    full = x / sin(x)
    assert not torch.any(torch.isinf(texpand)|torch.isnan(texpand)),'sincinv texpand inf'+torch.any(torch.isinf(texpand))
    return texpand, full

# Lie 群作用于 R3

# R3 上的 Hodge 星算子
def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1], 3, 3, **to(k))
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

# 逆 Hodge 星算子
def uncross_matrix(K):
    """Application of hodge star on R3, mapping Λ^2 R3 -> Λ^1 R3"""
    k = torch.zeros(*K.shape[:-1], **to(K))
    k[...,0] = (K[...,2,1] - K[...,1,2])/2
    k[...,1] = (K[...,0,2] - K[...,2,0])/2
    k[...,2] = (K[...,1,0] - K[...,0,1])/2
    return k

# SO3 类
class SO3:
    lie_dim = 3
    rep_dim = 3
    q_dim = 1

    def __init__(self, alpha = .2):
        super().__init__()
        self.alpha = alpha
    
    # 计算指数映射
    def exp(self,w):
        """ Computes (matrix) exponential Lie algebra elements (in a given basis).
            ie out = exp(\sum_i a_i A_i) where A_i are the exponential generators of G.
            Input: [a (*,lie_dim)] where * is arbitrarily shaped
            Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""

        """ Rodriguez's formula, assuming shape (*,3)
            where components 1,2,3 are the generators for xrot,yrot,zrot"""
        theta = w.norm(dim=-1)[..., None, None]
        K = cross_matrix(w)
        I = torch.eye(3, **to(K))
        Rs = I + K * sinc(theta) + (K @ K) * cosc(theta)
        return Rs
    
    # 计算对数映射
    def log(self,R):
        """ Computes components in terms of generators rx,ry,rz. Shape (*,3,3)"""

        """ Computes (matrix) logarithm for collection of matrices and converts to Lie algebra basis.
            Input [u (*,rep_dim,rep_dim)]
            Output [coeffs of log(u) in basis (*,d)] """
        trR = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        costheta = ((trR-1) / 2).clamp(max=1, min=-1).unsqueeze(-1)
        theta = acos(costheta)
        logR = uncross_matrix(R) * sinc_inv(theta)
        return logR

    # 计算逆元素
    def inv(self,g):
        """ We can compute the inverse of elements g (*,rep_dim,rep_dim) as exp(-log(g))"""
        return self.exp(-self.log(g))
    def elems2pairs(self,a):
        """ 计算输入中沿着 n 维度的所有 a b 对的 log(e^-b e^a)。
            输入: [a (bs,n,d)] 输出: [pairs_ab (bs,n,n,d)] """
        # 计算 e^-a 的逆
        vinv = self.exp(-a.unsqueeze(-3))
        # 计算 e^a
        u = self.exp(a.unsqueeze(-2))
        # 计算 log(e^-b e^a)
        return self.log(vinv@u)    # ((bs,1,n,d) -> (bs,1,n,r,r))@((bs,n,1,d) -> (bs,n,1,r,r))

    def lift(self, x, nsamples, **kwargs):
        """ 假设 p 的形状为 (*,n,2)，vals 的形状为 (*,n,c)，mask 的形状为 (*,n)
            返回形状为 [(*,n*nsamples,lie_dim),(*,n*nsamples,c)] 的 (a,v) """
        p, v, m, e = x
        # 将 p 展开为 (bs,n*ns,d) 和 (bs,n*ns,qd)
        expanded_a = self.lifted_elems(p,nsamples,**kwargs)
        nsamples = expanded_a.shape[-2]//m.shape[-1]
        # 将 v 和 mask 像 q 一样展开
        expanded_v = repeat(v, 'b n c -> b (n m) c', m = nsamples) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = repeat(m, 'b n -> b (n m)', m = nsamples) # (bs,n) -> (bs,n,ns) -> (bs,n*ns)
        expanded_e = repeat(e, 'b n1 n2 c -> b (n1 m1) (n2 m2) c', m1 = nsamples, m2 = nsamples) if exists(e) else None

        # 从 elems 转换为 pairs
        paired_a = self.elems2pairs(expanded_a) #(bs,n*ns,d) -> (bs,n*ns,n*ns,d)
        embedded_locations = paired_a
        return (embedded_locations,expanded_v,expanded_mask, expanded_e)
class SE3(SO3):
    # 定义 SE3 类，继承自 SO3 类
    lie_dim = 6
    # 定义李代数维度为 6
    rep_dim = 4
    # 定义表示维度为 4
    q_dim = 0
    # 定义 q 维度为 0

    def __init__(self, alpha=.2, per_point=True):
        # 初始化函数，接受 alpha 和 per_point 两个参数
        super().__init__()
        # 调用父类的初始化函数
        self.alpha = alpha
        # 设置对象的 alpha 属性为传入的 alpha 值
        self.per_point = per_point
        # 设置对象的 per_point 属性为传入的 per_point 值

    def exp(self,w):
        # 定义 exp 函数，接受参数 w
        dd_kwargs = to(w)
        # 将 w 转换为 dd_kwargs
        theta = w[...,:3].norm(dim=-1)[...,None,None]
        # 计算 w 的前三个元素的范数，并扩展维度
        K = cross_matrix(w[...,:3])
        # 计算 w 的前三个元素的叉乘矩阵
        R = super().exp(w[...,:3])
        # 调用父类的 exp 函数，计算 w 的前三个元素的指数映射
        I = torch.eye(3, **dd_kwargs)
        # 创建 3x3 的单位矩阵
        V = I + cosc(theta)*K + sincc(theta)*(K@K)
        # 计算 V 矩阵
        U = torch.zeros(*w.shape[:-1],4,4, **dd_kwargs)
        # 创建全零的 4x4 矩阵
        U[...,:3,:3] = R
        # 将 R 赋值给 U 的前三行前三列
        U[...,:3,3] = (V@w[...,3:].unsqueeze(-1)).squeeze(-1)
        # 计算并赋值 U 的前三行第四列
        U[...,3,3] = 1
        # 设置 U 的第四行第四列为 1
        return U
        # 返回 U 矩阵
    
    def log(self,U):
        # 定义 log 函数，接受参数 U
        w = super().log(U[..., :3, :3])
        # 调用父类的 log 函数，计算 U 的前三行前三列的对数映射
        I = torch.eye(3, **to(w))
        # 创建 3x3 的单位矩阵
        K = cross_matrix(w[..., :3])
        # 计算 w 的前三个元素的叉乘矩阵
        theta = w.norm(dim=-1)[..., None, None]#%(2*pi)
        # 计算 w 的范数，并扩展维度
        cosccc = coscc(theta)
        # 计算 coscc(theta)
        Vinv = I - K/2 + cosccc*(K@K)
        # 计算 Vinv 矩阵
        u = (Vinv @ U[..., :3, 3].unsqueeze(-1)).squeeze(-1)
        # 计算 u 向量
        return torch.cat([w, u], dim=-1)
        # 返回拼接后的 w 和 u 向量

    def lifted_elems(self,pt,nsamples):
        """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
            use a different group element per atom in the molecule"""
        # 返回 farthest_lift 函数的结果
        # same lifts for each point right now
        bs,n = pt.shape[:2]
        # 获取 pt 的形状
        dd_kwargs = to(pt)
        # 将 pt 转换为 dd_kwargs

        q = torch.randn(bs, (n if self.per_point else 1), nsamples, 4, **dd_kwargs)
        # 生成服从标准正态分布的随机数
        q /= q.norm(dim=-1).unsqueeze(-1)
        # 对 q 进行归一化

        theta_2 = atan2(q[..., 1:].norm(dim=-1),q[..., 0])[..., None]
        # 计算角度 theta_2
        so3_elem = 2 * sinc_inv(theta_2) * q[...,1:]
        # 计算 so3_elem
        se3_elem = torch.cat([so3_elem, torch.zeros_like(so3_elem)], dim=-1)
        # 拼接得到 se3_elem
        R = self.exp(se3_elem)
        # 计算 se3_elem 的指数映射

        T = torch.zeros(bs, n, nsamples, 4, 4, **dd_kwargs)
        # 创建全零的 4x4 矩阵
        T[..., :, :] = torch.eye(4, **dd_kwargs)
        # 将单位矩阵赋值给 T
        T[..., :3, 3] = pt[..., None, :]
        # 将 pt 赋值给 T 的前三行第四列

        a = self.log(T @ R)
        # 计算 T @ R 的对数映射
        return a.reshape(bs, n * nsamples, 6)
        # 返回重塑后的结果

    def distance(self,abq_pairs):
        # 定义 distance 函数，接受参数 abq_pairs
        dist_rot = abq_pairs[...,:3].norm(dim=-1)
        # 计算旋转部分的距离
        dist_trans = abq_pairs[...,3:].norm(dim=-1)
        # 计算平移部分的距离
        return dist_rot * self.alpha + (1-self.alpha) * dist_trans
        # 返回旋转部分距禂乘以 alpha 加上平移部分距离乘以 (1-alpha) 的结果
```