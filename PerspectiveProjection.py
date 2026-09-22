import os
import csv
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml


class _FlowList(list):
    """标记为 flow style 的 list，dump 时渲染成 [a, b, c] 形式而不是逐行的 - 列表。"""
    pass


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(_FlowList, _flow_list_representer)

board_pattern=(10,6)   # 1000那张图
#board_pattern=(8,11)   # 003那张图
#board_pattern=(3,3)   #IDEA 缺白边，去畸变时getOptimalNewCameraMatrix改为保留全部视野试试？
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def _chessboard_orientations(corners, pattern):
    """棋盘检测可能的 4 种角点顺序（起点在四个角）。"""
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    return [
        grid.reshape(-1, 2),
        grid[:, ::-1, :].reshape(-1, 2),
        grid[::-1, :, :].reshape(-1, 2),
        grid[::-1, ::-1, :].reshape(-1, 2),
    ]


def align_corners(ref_corners, corners, pattern):
    """按相邻帧对齐：选与上一帧对应点平均距离最小的朝向。
    相邻帧位移小，顺序错 180° 时距离会到棋盘对角线量级，可消掉相似变换的歧义。
    """
    best, best_err = corners, np.inf
    ref = ref_corners.astype(np.float64).reshape(-1, 2)
    for cand in _chessboard_orientations(corners, pattern):
        err = np.mean(np.linalg.norm(cand.astype(np.float64) - ref, axis=1))
        if err < best_err:
            best_err, best = err, cand
    return best


def find_coners(image_path, show=True, ref_corners=None):
    image=cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None

    if image.ndim == 3:
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayImg = image

    ret, corners = cv2.findChessboardCorners(grayImg,board_pattern, None)
    if not ret:
        return None
    corners = cv2.cornerSubPix(grayImg, corners, (11,11), (-1,-1), criteria)
    corners = corners.reshape(-1, 2)
    if ref_corners is not None:
        corners = align_corners(ref_corners, corners, board_pattern)
    if show:
        cv2.drawChessboardCorners(image, board_pattern, corners, ret)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return corners


def _frontal_dst_points(corners, pattern):
    """用本帧棋盘生成正视网格，只用于估计图像→iPad 平面的 H。
    dest 跟着本帧棋盘朝向；后续帧应复用这份 H，而不是每帧重做 dest。
    """
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    dx = np.mean(np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=-1))
    dy = np.mean(np.linalg.norm(grid[1:] - grid[:-1], axis=-1))
    square_size = float((dx + dy) / 2.0)

    mean_row = (grid[:, 1:] - grid[:, :-1]).mean(axis=(0, 1))
    mean_col = (grid[1:] - grid[:-1]).mean(axis=(0, 1))
    angle = np.arctan2(mean_row[1], mean_row[0])

    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    local = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) * square_size
    local_center = local.mean(axis=0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    # 旋转后的 +y 应与原图列方向同侧，只防镜像，不把图摆正
    rot_y = np.array([-sin_a, cos_a], dtype=np.float32)
    if np.dot(rot_y, mean_col) < 0:
        local[:, 1] = 2.0 * local_center[1] - local[:, 1]
    dst = (local - local_center) @ R.T + corners.mean(axis=0)
    return dst, square_size


def _homography_from_corners(src_pts, image_shape):
    """用本帧角点估计图像→正视平面的单应性，并算出画布平移与尺寸。"""
    dst_pts, _ = _frontal_dst_points(src_pts, board_pattern)
    src_quad = _outer_quad(src_pts, board_pattern)
    dst_quad = _outer_quad(dst_pts, board_pattern)
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)
    if H is None:
        return None, None
    h, w = image_shape[:2]
    img_corners = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
    warped_corners = cv2.perspectiveTransform(img_corners, H).reshape(-1, 2)
    xmin, ymin = warped_corners.min(axis=0)
    xmax, ymax = warped_corners.max(axis=0)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    H_full = T @ H
    out_w = int(np.ceil(xmax - xmin))
    out_h = int(np.ceil(ymax - ymin))
    max_side = 3000
    scale = min(1.0, max_side / max(out_w, out_h))
    if scale < 1.0:
        H_full = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64) @ H_full
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)
    return H_full, (out_w, out_h)


def perspetive_projection(image_path, show=True, ref_corners=None, H_full=None, out_size=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None, None, None, None, None

    # 获取原图上的角点，估计棋盘平面到正视平面的单应性
    corners_list = find_coners(image_path, show=show, ref_corners=ref_corners)
    if corners_list is None or len(corners_list) == 0:
        print("未找到棋盘角点，无法透视变换")
        return None, None, None, None, None

    src_pts = corners_list.astype(np.float32)
    # iPad/桌面不动，只有屏幕内容在转：H 应对准这块固定平面。
    # 传入已有 H 则本帧只 warp，避免 dest 跟着棋盘转而把 iPad 拧转。
    if H_full is None:
        H_full, out_size = _homography_from_corners(src_pts, image.shape)
        if H_full is None:
            print("单应性估计失败")
            return None, None, None, None, None

    out_w, out_h = out_size
    projected_image = cv2.warpPerspective(image, H_full, (out_w, out_h))
    if show:
        cv2.imshow("projected", projected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return projected_image, image, src_pts, H_full, out_size


def _outer_quad(corners, pattern):
    """棋盘内角点网格的外围 4 点，按周向排列，避免折线交叉。"""
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    return np.array([
        grid[0, 0],
        grid[0, -1],
        grid[-1, -1],
        grid[-1, 0],
    ], dtype=np.float32)


def _interior_angles_deg(quad):
    """凸四边形各顶点内夹角，单位 degree，按顶点周向顺序返回。"""
    pts = np.asarray(quad, dtype=np.float64)
    n = len(pts)
    angles = np.empty(n, dtype=np.float64)
    for i in range(n):
        v1 = pts[(i - 1) % n] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        cos_a = np.dot(v1, v2) / (n1 * n2)
        angles[i] = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return angles


def _draw_quad(image, quad, color=(0, 0, 255)):
    vis = image.copy()
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
    angles = _interior_angles_deg(quad)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 8, color, -1)
        prev_pt = quad[(i - 1) % 4]
        next_pt = quad[(i + 1) % 4]
        curr_pt = quad[i]
        u1 = prev_pt - curr_pt
        u2 = next_pt - curr_pt
        u1 = u1 / (np.linalg.norm(u1) + 1e-12)
        u2 = u2 / (np.linalg.norm(u2) + 1e-12)
        bisector = u1 + u2
        bisector = bisector / (np.linalg.norm(bisector) + 1e-12)
        label = f"{angles[i]:.1f} deg"
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        tx, ty = curr_pt + bisector * 48.0
        org = (int(tx - tw / 2), int(ty + th / 2))
        org = (max(0, min(org[0], vis.shape[1] - tw)),
               max(th, min(org[1], vis.shape[0] - 1)))
        cv2.putText(vis, label, org, font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, org, font, 0.55, (180, 105, 255), 1, cv2.LINE_AA)
    return vis


def _hconcat_resize(img_left, img_right):
    h = max(img_left.shape[0], img_right.shape[0])
    def _to_h(img):
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))
    return cv2.hconcat([_to_h(img_left), _to_h(img_right)])


def draw_perspectiveTransform(image, projected_image, corners, H_full,
                              pattern=board_pattern, show=True,
                              save_path=r"./output/perspectiveTransform.jpg"):
    """在原图与投影图上描出棋盘外围 4 点，横向拼接保存。
    返回 (translation, proj_angles)：
      translation  projected_image 相对原图的平移量 (dx, dy)
      proj_angles  投影图上外围四边形 4 个顶点的内夹角，单位 degree
    """
    src_quad = _outer_quad(corners, pattern)
    proj_quad = cv2.perspectiveTransform(src_quad.reshape(-1, 1, 2), H_full).reshape(-1, 2)
    translation = proj_quad.mean(axis=0) - src_quad.mean(axis=0)
    proj_angles = _interior_angles_deg(proj_quad)

    vis_src = _draw_quad(image, src_quad)
    vis_proj = _draw_quad(projected_image, proj_quad)
    concat = _hconcat_resize(vis_src, vis_proj)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    stem, ext = os.path.splitext(save_path)
    src_path = f"{stem}_src{ext}"
    proj_path = f"{stem}_proj{ext}"
    proj_raw_path = f"{stem}_proj_raw{ext}"
    cv2.imwrite(save_path, concat)
    cv2.imwrite(src_path, vis_src)
    cv2.imwrite(proj_path, vis_proj)
    cv2.imwrite(proj_raw_path, projected_image)
    print(f"已保存: {save_path}")
    print(f"已保存: {src_path}")
    print(f"已保存: {proj_path}")
    print(f"已保存: {proj_raw_path}")

    if show:
        cv2.imshow("perspectiveTransform", concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return translation, proj_angles


def scan_max_wh(img_folder, img_list):
    """读取列表中所有图片的长宽，返回 (Hmax, Wmax)。"""
    h_max, w_max = 0, 0
    for img_name in img_list:
        img = cv2.imread(os.path.join(img_folder, img_name))
        if img is None:
            continue
        h, w = img.shape[:2]
        h_max = max(h_max, h)
        w_max = max(w_max, w)
    return h_max, w_max


def paste_aligned_canvas(proj_img, proj_p0, ref_p0, canvas_h, canvas_w):
    """把投影图平移，使 proj_p0 落到 ref_p0，再裁切/补全到 (canvas_h, canvas_w)。"""
    ox = int(round(float(ref_p0[0] - proj_p0[0])))
    oy = int(round(float(ref_p0[1] - proj_p0[1])))
    channels = 1 if proj_img.ndim == 2 else proj_img.shape[2]
    if proj_img.ndim == 2:
        canvas = np.zeros((canvas_h, canvas_w), dtype=proj_img.dtype)
    else:
        canvas = np.zeros((canvas_h, canvas_w, channels), dtype=proj_img.dtype)
    ph, pw = proj_img.shape[:2]
    dx0, dy0 = ox, oy
    dx1, dy1 = ox + pw, oy + ph
    cx0, cy0 = max(0, dx0), max(0, dy0)
    cx1, cy1 = min(canvas_w, dx1), min(canvas_h, dy1)
    if cx0 >= cx1 or cy0 >= cy1:
        return canvas
    sx0, sy0 = cx0 - dx0, cy0 - dy0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    canvas[cy0:cy1, cx0:cx1] = proj_img[sy0:sy1, sx0:sx1]
    return canvas


def save_homography_yaml(H, out_size, img_folder, yaml_path):
    """把单应矩阵及相关信息保存到 yaml 文件。

    记录内容：
      script_path  当前运行的 .py 文件绝对路径
      save_date    保存日期时间
      img_folder   本次处理的图片文件夹绝对路径
      out_size     单应变换对应的输出画布尺寸
      homography   3x3 单应矩阵（按行展开的嵌套列表）
    """
    H_rows = np.asarray(H, dtype=np.float64).tolist()
    data = {
        "script_path": os.path.abspath(__file__),
        "save_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "img_folder": os.path.abspath(img_folder),
        "out_size": {"width": int(out_size[0]), "height": int(out_size[1])},
        "homography": [_FlowList(row) for row in H_rows],
    }
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"已保存: {yaml_path}")


def save_angles_csv(rows, csv_path):
    """rows: list of (img_name, angle1, angle2, angle3, angle4)，保存至 2 位小数。"""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "angle1", "angle2", "angle3", "angle4"])
        for img_name, angles in rows:
            writer.writerow([img_name] + [f"{a:.2f}" for a in angles])
    print(f"已保存: {csv_path}")


def plot_angles(rows, plot_path):
    """rows: list of (img_name, angle1..angle4)，每个角度画一条线，共 4 条。"""
    if not rows:
        print("无数据可画图")
        return
    img_names = [r[0] for r in rows]
    angles_arr = np.array([r[1] for r in rows], dtype=np.float64)  # (N, 4)
    x = np.arange(len(img_names))

    plt.figure(figsize=(max(8, len(img_names) * 0.3), 6))
    labels = ["angle1", "angle2", "angle3", "angle4"]
    for i in range(4):
        plt.plot(x, angles_arr[:, i], marker="o", label=labels[i])
    plt.xticks(x, img_names, rotation=90, fontsize=6)
    plt.xlabel("image")
    plt.ylabel("angle (deg)")
    plt.title("Projected outer-quad angles")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"已保存: {plot_path}")

if __name__ == "__main__":
    #image_path=r"./data/test_undist.jpg"
    #image_path=r"./data/003_20260525_042258_495374.png"

    img_folder = r"./data/RotatedChessBoard_renamed"
    output_folder = r"./output/folder"

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(img_exts)]
    img_list.sort()

    if not img_list:
        print(f"未在文件夹中找到图片: {img_folder}")

    h_max, w_max = scan_max_wh(img_folder, img_list)
    print(f"Hmax={h_max}, Wmax={w_max}")

    angle_rows = []  # list of (img_name, angles(4,))
    ref_corners = None
    shared_H = None
    shared_size = None
    for img_name in img_list:
        image_path = os.path.join(img_folder, img_name)
        print(f"正在处理: {image_path}")
        stem, ext = os.path.splitext(img_name)
        save_path = os.path.join(output_folder, f"{stem}_perspectiveTransform.jpg")

        projected_image, image, src_pts, H_full, out_size = perspetive_projection(
            image_path, show=False, ref_corners=ref_corners,
            H_full=shared_H, out_size=shared_size)
        if projected_image is None:
            print(f"跳过: {image_path}")
            continue
        ref_corners = src_pts
        if shared_H is None:
            shared_H, shared_size = H_full, out_size
            print("已固定首帧单应性(iPad平面)，后续帧共用，避免平板被拧转")
            homography_yaml_path = os.path.join(output_folder, "homography.yaml")
            save_homography_yaml(shared_H, shared_size, img_folder, homography_yaml_path)
        # 同一 H 下 iPad 已对齐，不再按棋盘角点平移（否则平板会跟着视频内容挪）
        aligned = paste_aligned_canvas(
            projected_image, (0.0, 0.0), (0.0, 0.0), h_max, w_max)
        aligned_path = os.path.join(output_folder, f"{stem}_aligned{ext}")
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(aligned_path, aligned)
        print(f"已保存: {aligned_path}")

        translation, proj_angles = draw_perspectiveTransform(
            image, projected_image, src_pts, H_full, show=False, save_path=save_path)
        print(f"平移量: dx={translation[0]:.2f}, dy={translation[1]:.2f}")
        angle_rows.append((img_name, proj_angles))

    csv_path = os.path.join(output_folder, "angles.csv")
    save_angles_csv(angle_rows, csv_path)

    plot_path = os.path.join(output_folder, "angles_plot.png")
    plot_angles(angle_rows, plot_path)
    print(f"已保存: {homography_yaml_path}")
    print("Completed!")
