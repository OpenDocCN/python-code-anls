# `.\yolov8\ultralytics\trackers\utils\kalman_filter.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import numpy as np
import scipy.linalg


class KalmanFilterXYAH:
    """
    For bytetrack. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, a, h, vx, vy, va, vh) contains the bounding box center position (x, y), aspect
    ratio a, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, a, h) is taken as direct
    observation of the state space (linear observation model).
    """

    def __init__(self):
        """Initialize Kalman filter model matrices with motion and observation uncertainty weights."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        # 初始化 Kalman 滤波器的运动模型矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # 观测矩阵，将状态空间映射到观测空间
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. These weights control
        # the amount of uncertainty in the model.
        # 运动和观测的不确定性权重，相对于当前状态估计来选择
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a,
                and height h.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        # 使用未关联的测量创建跟踪轨迹

        # 初始位置为测量值
        mean_pos = measurement
        # 初始速度为零向量
        mean_vel = np.zeros_like(mean_pos)
        # 组合成完整的状态向量（位置和速度）
        mean = np.r_[mean_pos, mean_vel]

        # 根据测量的不确定性，设置协方差矩阵的标准差
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        # 根据标准差创建对角协方差矩阵
        covariance = np.diag(np.square(std))
        # 返回初始化的状态向量和协方差矩阵
        return mean, covariance
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        # 根据位置权重和对象速度初始化预测步骤中的位置标准偏差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        # 根据速度权重和对象速度初始化预测步骤中的速度标准偏差
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        # 计算运动噪声协方差矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 预测对象状态的均值向量，使用运动矩阵进行状态预测
        mean = np.dot(mean, self._motion_mat.T)
        # 预测对象状态的协方差矩阵，考虑运动和运动噪声的影响
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        # 根据位置权重和对象速度初始化测量空间中的标准偏差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        # 计算创新协方差矩阵，用于测量预测和实际测量之间的差异
        innovation_cov = np.diag(np.square(std))

        # 将状态分布投影到测量空间中的均值向量，使用更新矩阵进行投影
        mean = np.dot(self._update_mat, mean)
        # 将状态分布投影到测量空间中的协方差矩阵，考虑创新噪声的影响
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov
    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        # Calculate standard deviations for position and velocity components
        std_pos = [
            self._std_weight_position * mean[:, 3],     # Standard deviation for x position
            self._std_weight_position * mean[:, 3],     # Standard deviation for y position
            1e-2 * np.ones_like(mean[:, 3]),            # Small value for aspect ratio 'a'
            self._std_weight_position * mean[:, 3],     # Standard deviation for height 'h'
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],     # Standard deviation for x velocity
            self._std_weight_velocity * mean[:, 3],     # Standard deviation for y velocity
            1e-5 * np.ones_like(mean[:, 3]),           # Small value for velocity in 'a'
            self._std_weight_velocity * mean[:, 3],     # Standard deviation for velocity in 'h'
        ]

        # Stack standard deviations and square them
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # Create motion covariance matrices
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        # Predict new mean using motion model
        mean = np.dot(mean, self._motion_mat.T)

        # Calculate left multiplication for covariance update
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))

        # Update covariance matrix incorporating motion and motion_cov
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h), where (x, y) is the center
                position, a the aspect ratio, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        # Project predicted mean and covariance based on measurement model
        projected_mean, projected_cov = self.project(mean, covariance)

        # Compute Cholesky factorization of projected_cov
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)

        # Compute Kalman gain using Cholesky factor and update matrix
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T

        # Compute innovation (difference between measurement and projected mean)
        innovation = measurement - projected_mean

        # Update mean and covariance using Kalman gain and innovation
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """
        计算状态分布与测量之间的门控距离。适当的距离阈值可以从 `chi2inv95` 获得。如果 `only_position` 是 False，
        则卡方分布的自由度为4，否则为2。

        Args:
            mean (ndarray): 状态分布的均值向量（8维）。
            covariance (ndarray): 状态分布的协方差（8x8维）。
            measurements (ndarray): 包含N个测量值的矩阵，每个测量值格式为 (x, y, a, h)，其中 (x, y) 是边界框的中心位置，
                a 是长宽比，h 是高度。
            only_position (bool, optional): 如果为 True，则仅基于边界框中心位置计算距离。默认为 False。
            metric (str, optional): 用于计算距离的度量标准。选项有 'gaussian' 表示平方欧氏距离，'maha' 表示平方马氏距离。
                默认为 'maha'。

        Returns:
            (np.ndarray): 返回长度为N的数组，其中第i个元素包含 (mean, covariance) 与 `measurements[i]` 之间的平方距离。
        """
        # 将均值和协方差投影到合适的维度
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            # 如果只计算位置，则仅保留均值和协方差的前两个维度
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # 计算测量值与均值之间的差异
        d = measurements - mean
        
        if metric == "gaussian":
            # 计算平方欧氏距离
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            # 计算平方马氏距离
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # 平方马氏距离
        else:
            raise ValueError("无效的距离度量标准")
class KalmanFilterXYWH(KalmanFilterXYAH):
    """
    For BoT-SORT. A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains the bounding box center position (x, y), width
    w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).
    """

    def initiate(self, measurement: np.ndarray) -> tuple:
        """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
        # 使用测量值初始化均值向量，包括位置和速度
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # 计算测量值的标准差，用于构建协方差矩阵
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        # 计算位置和速度的预测标准差
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 预测下一个时刻的均值和协方差矩阵
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    def project(self, mean, covariance) -> tuple:
        """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
        # Calculate standard deviations for position components based on mean values
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        # Create innovation covariance matrix from computed standard deviations
        innovation_cov = np.diag(np.square(std))

        # Project the mean vector using the update matrix
        mean = np.dot(self._update_mat, mean)
        # Project the covariance matrix using the update matrix
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        # Return the projected mean and adjusted covariance matrix
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance) -> tuple:
        """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
        # Compute standard deviations for position and velocity components
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
        ]
        # Stack position and velocity standard deviations and compute their squares
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # Construct motion covariance matrices for each object in the prediction
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        # Predict mean vector using motion matrix
        mean = np.dot(mean, self._motion_mat.T)
        # Predict covariance matrix incorporating motion covariance
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        # Return predicted mean and covariance matrix
        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple:
        """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
        # Delegate the update step to the superclass, passing mean, covariance, and measurement
        return super().update(mean, covariance, measurement)
```