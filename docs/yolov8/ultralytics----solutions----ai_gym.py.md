# `.\yolov8\ultralytics\solutions\ai_gym.py`

```py
# 导入OpenCV库，用于图像处理
import cv2

# 导入自定义函数和类
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator

# AIGym类用于实时视频流中人员姿势的管理
class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(
        self,
        kpts_to_check,
        line_thickness=2,
        view_img=False,
        pose_up_angle=145.0,
        pose_down_angle=90.0,
        pose_type="pullup",
    ):
        """
        Initializes the AIGym class with the specified parameters.

        Args:
            kpts_to_check (list): Indices of keypoints to check.
            line_thickness (int, optional): Thickness of the lines drawn. Defaults to 2.
            view_img (bool, optional): Flag to display the image. Defaults to False.
            pose_up_angle (float, optional): Angle threshold for the 'up' pose. Defaults to 145.0.
            pose_down_angle (float, optional): Angle threshold for the 'down' pose. Defaults to 90.0.
            pose_type (str, optional): Type of pose to detect ('pullup', 'pushup', 'abworkout'). Defaults to "pullup".
        """

        # 图像和线条厚度
        self.im0 = None  # 初始图像设为None
        self.tf = line_thickness  # 线条厚度设定为传入的参数值

        # 关键点和计数信息
        self.keypoints = None  # 关键点初始化为None
        self.poseup_angle = pose_up_angle  # 'up'姿势的角度阈值
        self.posedown_angle = pose_down_angle  # 'down'姿势的角度阈值
        self.threshold = 0.001  # 阈值设定为0.001

        # 存储阶段、计数和角度信息
        self.angle = None  # 角度信息初始化为None
        self.count = None  # 计数信息初始化为None
        self.stage = None  # 阶段信息初始化为None
        self.pose_type = pose_type  # 姿势类型，默认为"pullup"
        self.kpts_to_check = kpts_to_check  # 需要检查的关键点索引列表

        # 可视化信息
        self.view_img = view_img  # 是否显示图像的标志
        self.annotator = None  # 标注器初始化为None

        # 检查环境是否支持imshow函数
        self.env_check = check_imshow(warn=True)  # 调用自定义函数检查环境支持情况
        self.count = []  # 计数列表初始化为空列表
        self.angle = []  # 角度列表初始化为空列表
        self.stage = []  # 阶段列表初始化为空列表
    def start_counting(self, im0, results):
        """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data.
        """

        # 将当前帧图像保存到对象的属性中
        self.im0 = im0

        # 如果没有检测到姿态估计数据，则直接返回原始图像
        if not len(results[0]):
            return self.im0

        # 如果检测到的人数超过已记录的计数器数量，进行扩展
        if len(results[0]) > len(self.count):
            new_human = len(results[0]) - len(self.count)
            self.count += [0] * new_human
            self.angle += [0] * new_human
            self.stage += ["-"] * new_human

        # 获取关键点数据
        self.keypoints = results[0].keypoints.data
        # 创建一个用于绘制的注释器对象
        self.annotator = Annotator(im0, line_width=self.tf)

        # 遍历检测到的关键点数据
        for ind, k in enumerate(reversed(self.keypoints)):
            # 估算姿势角度并绘制特定关键点
            if self.pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                self.angle[ind] = self.annotator.estimate_pose_angle(
                    k[int(self.kpts_to_check[0])].cpu(),
                    k[int(self.kpts_to_check[1])].cpu(),
                    k[int(self.kpts_to_check[2])].cpu(),
                )
                # 在图像上绘制指定关键点
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)

                # 根据角度更新姿势阶段和计数
                if self.pose_type in {"abworkout", "pullup"}:
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "down"
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "down":
                        self.stage[ind] = "up"
                        self.count[ind] += 1

                elif self.pose_type in {"pushup", "squat"}:
                    if self.angle[ind] > self.poseup_angle:
                        self.stage[ind] = "up"
                    if self.angle[ind] < self.posedown_angle and self.stage[ind] == "up":
                        self.stage[ind] = "down"
                        self.count[ind] += 1

                # 绘制角度、计数和姿势阶段的信息
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],
                    count_text=self.count[ind],
                    stage_text=self.stage[ind],
                    center_kpt=k[int(self.kpts_to_check[1])],
                )

            # 绘制关键点
            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        # 如果环境支持并且需要显示图像，则显示处理后的图像
        if self.env_check and self.view_img:
            cv2.imshow("Ultralytics YOLOv8 AI GYM", self.im0)
            # 等待用户按键以退出显示
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

        # 返回处理后的图像
        return self.im0
# 如果当前脚本作为主程序运行（而非被导入其他模块），执行以下代码块
if __name__ == "__main__":
    # 定义一个示例的关键点列表，用于检查
    kpts_to_check = [0, 1, 2]  # example keypoints
    # 创建一个 AIGym 对象，并传入关键点列表作为参数
    aigym = AIGym(kpts_to_check)
```