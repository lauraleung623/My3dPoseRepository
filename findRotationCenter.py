import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from frompic2video import generate_video

# TODO 加入RANSAC、自动剔除异常角点
# TODO 另外保存一个放大版的图片，用于显示误差

board_pattern = (10, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def _chessboard_orientations(corners, pattern):
    """棋盘检测可能的 4 种角点顺序（起点在四个角）。"""
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    return [
        grid.reshape(-1, 2), # 原序
        grid[:, ::-1, :].reshape(-1, 2), # 左右翻转
        grid[::-1, :, :].reshape(-1, 2), # 上下翻转
        grid[::-1, ::-1, :].reshape(-1, 2), # 对角翻转
    ]


def align_corners(ref_corners, corners, pattern):
    """
    把当前帧角点重排到与参考帧同一套物理点顺序。
    用相似变换拟合误差选择 4 种棋盘朝向中最好的一种。
    """
    best, best_err = corners, np.inf
    ref = ref_corners.astype(np.float64)
    ones = np.ones((len(ref), 1))
    for cand in _chessboard_orientations(corners, pattern): # 历遍4种棋盘朝向
        M, _ = cv2.estimateAffinePartial2D(ref, cand.astype(np.float64)) #评估一个相似变换
        if M is None:
            continue
        pred = (M @ np.hstack([ref, ones]).T).T
        err = np.mean(np.linalg.norm(pred - cand, axis=1))
        if err < best_err: # 对应关系对时，整块棋盘格可以用一个平面相似变换相似，得到较小误差；对应关系错时，一个相似变换兜不住60个点
            best_err, best = err, cand  # 使用相似变换最小的那套平移反转
    return best


def find_coners(image_folder):
    """
    返回值数据格式(N,M,2)，N是帧数，M是角点数，2是坐标
    """
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(image_folder) if f.lower().endswith(img_exts)]
    img_list.sort()

    coners_list = []
    ref_corners = None
    for image_path in img_list:
        image = cv2.imread(os.path.join(image_folder, image_path))
        if image is None:
            print(f"无法读取图片: {image_path}")
            continue

        ret, corners = cv2.findChessboardCorners(image, board_pattern, None)
        if ret:
            corners = corners.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)
            grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(grayImg, corners, (11,11), (-1,-1), criteria) # 每帧的坐标
            corners = corners.reshape(-1, 2)
            if ref_corners is None:
                ref_corners = corners
            else:
                corners = align_corners(ref_corners, corners, board_pattern)
            corners = corners[np.newaxis, ...]  # 新增帧维度
            if len(coners_list) == 0:
                coners_list = corners
            else:
                coners_list = np.concatenate((coners_list, corners), axis=0)
    return coners_list

def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx = -D / 2
    cy = -E / 2
    r = np.sqrt(cx**2 + cy**2 - F)
    return cx, cy, r

def fit_circles(points):
    """
    points: 三维数组, shape = (M, N, 2)
    沿第0维遍历，对每个 (N, 2) 的切片调用 fit_circle 拟合圆心与半径，
    返回长度为 M 的列表，每个元素为 (cx, cy, r)
    """
    results = []
    #for i in range(points.shape[0]): # 错的
    for i in range(points.shape[1]): # 历遍左右角点
        cx, cy, r = fit_circle(points[:,i,:])   # 提取同一个点，在每帧的位置
        results.append((cx, cy, r))
    return results

def estimate_center_joint(points, mode="median", filter=False):
    """
    使用一系列拟合的圆心，估计旋转中心
    :param points: (N, 2) 或 (N, 3)，每行至少包含 (x, y)
    :param mode: "median" 分别对 x、y 求中位数；"centroid" 求这一系列点的重心
    :param filter: True 时先用 MAD 剔除异常点，再估计中心
    """
    xy = np.asarray(points)[:, :2]

    # MAD剔除异常点，数据预处理，默认不执行
    if filter:
        xy_med = np.median(xy, axis=0)
        dist = np.linalg.norm(xy - xy_med, axis=1)
        mad = np.median(dist)
        if mad > 1e-12:
            inliers = xy[dist <= 3.0 * 1.4826 * mad]
            if len(inliers) > 0:
                xy = inliers

    if mode == "median":
        center_joint = np.median(xy, axis=0)
    elif mode == "centroid":
        center_joint = np.mean(xy, axis=0)
    else:
        raise ValueError(f"未知 mode: {mode}，应为 'median' 或 'centroid'")

    return center_joint

def draw_cross(image, center, color, size=50, thickness=2):
    """
    在 image 上以 center 为中心画一个十字
    若 center 为 None，则不画（直接返回）
    :param image: 图像 (会被直接修改)
    :param center: (x, y) 坐标，可为 None
    :param color: BGR颜色, 如 (255, 0, 0) 蓝色, (0, 255, 0) 绿色
    :param size: 十字臂长
    :param thickness: 线宽
    """
    if center is None:
        return
    x, y = int(center[0]), int(center[1])
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)

def draw_legend(image, items, top_left=(10, 10), box_size=20, gap=10,
                 font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2):
    """
    在 image 左上角画图例
    :param image: 图像 (会被直接修改)
    :param items: [(color, text), ...] 颜色为 BGR
    :param top_left: 图例起始坐标 (x, y)
    :param box_size: 颜色方块边长
    :param gap: 每行之间的间隔
    """
    x0, y0 = top_left
    line_height = box_size + gap
    for i, (color, text) in enumerate(items):
        y = y0 + i * line_height
        cv2.rectangle(image, (x0, y), (x0 + box_size, y + box_size), color, -1)
        cv2.putText(image, text, (x0 + box_size + 10, y + box_size - 3),
                    font, font_scale, color, thickness)

def visualize_center(video_path, output_path, center_detect, center_true=None, fps=10):
    """
    读取 video_path 视频，在每一帧上：
    - 用蓝色在 center_detect 坐标处画十字
    - 用绿色在 center_true 坐标处画十字
    并将所有画好十字的帧保存为 mp4 视频到 output_path

    :param video_path: 输入视频路径
    :param center_detect: (x, y) 检测中心坐标
    :param center_true: (x, y) 真实中心坐标
    :param output_path: 输出 mp4 视频路径
    :param fps: 输出视频帧率，默认 10
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if center_true is not None:
        error_dist = np.sqrt((center_detect[0] - center_true[0]) ** 2 +
                              (center_detect[1] - center_true[1]) ** 2)
    else:
        error_dist = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        draw_cross(frame, center_detect, (255, 0, 0))   # 蓝色: 检测中心
        draw_cross(frame, center_true, (0, 255, 0))      # 绿色: 真实中心
        legend_items = [((255, 0, 0), "Detected Center")]
        if center_true is not None:
            legend_items.append(((0, 255, 0), "True Center"))
        draw_legend(frame, legend_items, top_left=(100, 40),
                    box_size=50, gap=20, font_scale=2, thickness=3)
        if error_dist is not None:
            cv2.putText(frame, f"Error: {error_dist:.2f}pixels", (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"已保存视频: {output_path}")
    return

def visualize_center_error(center_detect, center_true, output_path):
    """
    计算 center_detect 各点相对 center_true 的误差距离，
    以折线图形式画出误差随点序号的变化曲线，
    并统计误差 std 和总点数，写在图片左上角，
    最终保存到 output_path 下，文件名为 "Error curve.png"

    :param center_detect: 长度为 N 的列表/数组，每个元素为 (cx, cy) 或 (cx, cy, r)
    :param center_true: (x, y) 真实中心坐标
    :param output_path: 保存图片的文件夹路径
    """
    # (N, 3) 的 (cx, cy, r) 只取每行前两列 (cx, cy)
    points = np.asarray(center_detect)[:, :2]
    n = len(points)

    dists = np.sqrt((points[:, 0] - center_true[0]) ** 2 +
                     (points[:, 1] - center_true[1]) ** 2)
    std = np.std(dists)

    plt.figure(figsize=(8, 6))
    # 仅画误差值随点序号变化的曲线
    plt.plot(np.arange(n), dists, '-o', color='red',
              linewidth=1, markersize=4, label='Error(pixel)')

    plt.xlabel('Point Index')
    plt.ylabel('Error (pixels)')
    plt.title('Error curve(abs value)')
    plt.legend()

    plt.text(0.02, 0.98, f"Std: {std:.3f}\nTotal points: {n}",
              transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "Error curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"已保存误差曲线图: {save_path}")
    return

if __name__ == "__main__":
    image_folder = r"./Rotation_Img/Center1367_1556"

    corner_list = find_coners(image_folder)  # 识别棋盘格个点
    print(f"corner_list.shape: {corner_list.shape}")
    circle_list = fit_circles(corner_list)   # 用格点拟合圆心
    print(f"circle_list.len: {len(circle_list)}") # 检测值的中心，是拟合值
    center_detect = estimate_center_joint(circle_list)   # 误差最小的旋转中心

    video_origin_path = r"./output/Rotation_Center.mp4"
    generate_video(image_folder, video_origin_path, fps=2)   # 生成原始视频
    
    video_cross_path = r"./output/Rotation_Center_cross.mp4"
    center_true = (1367.00,1556.00) # 模拟值的中心，是真值
    #center_detect = (circle_list[0][0], circle_list[0][1]) # 取第一个值计算
    visualize_center(video_origin_path, video_cross_path, center_detect, center_true, fps=2) # 可是化计算误差
    visualize_center_error(circle_list, center_true, r"./output") # 可视化plot出误差
