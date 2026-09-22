import os
import cv2
import numpy as np
import yaml


def load_homography_yaml(yaml_path):
    """读取 PerspectiveProjection_v7.py 保存的 homography.yaml，返回 3x3 numpy 矩阵。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return np.asarray(data["homography"], dtype=np.float64)


def _to_H_matrix(H):
    """把入参 H（list/tuple/np.ndarray，3x3，格式同 PerspectiveProjection_v7.py
    保存到 yaml 里的 homography 字段：[[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]）
    解析成 3x3 numpy 矩阵。
    """
    H_mat = np.asarray(H, dtype=np.float64)
    if H_mat.shape != (3, 3):
        raise ValueError(f"单应矩阵形状应为 (3,3)，实际为 {H_mat.shape}")
    return H_mat


def back_project_point(H, image_path, point, show=True):
    """根据单应矩阵 H，把投影图上一点反投影回原图坐标，并在原图拷贝上画出该点。

    参数：
      H           单应矩阵，3x3，格式与 PerspectiveProjection_v7.py 保存到 yaml 中的
                  homography 字段一致（嵌套 list/tuple/np.ndarray 均可）。
                  H 是 PerspectiveProjection_v7.py 中"原图 -> 投影图"方向的单应性
                  （perspetive_projection 用它把原图 warp 成投影图），
                  因此反投影（投影图 -> 原图）需要用其逆矩阵 H^-1。
      image_path  原图路径。
      point       投影图上某点坐标 (x, y)。
      show        是否显示效果图，默认 True。

    返回：
      Pmapping    该点在原图上对应的坐标 (x, y)，float 元组。
      vis_image   在原图拷贝上标出 Pmapping 的图像（BGR，np.ndarray）。
    """
    H_mat = _to_H_matrix(H)
    H_inv = np.linalg.inv(H_mat)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    pt = np.array([[point[0], point[1]]], dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pt, H_inv).reshape(-1, 2)[0]
    Pmapping = (float(mapped[0]), float(mapped[1]))

    vis_image = image.copy()
    px, py = int(round(Pmapping[0])), int(round(Pmapping[1]))
    color = (0, 0, 255)
    cv2.drawMarker(vis_image, (px, py), color, markerType=cv2.MARKER_CROSS,
                    markerSize=25, thickness=2)
    cv2.circle(vis_image, (px, py), 8, color, 2)
    label = f"({Pmapping[0]:.2f}, {Pmapping[1]:.2f})"
    cv2.putText(vis_image, label, (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis_image, label, (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    if show:
        cv2.imshow("BackProjection", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return Pmapping, vis_image


if __name__ == "__main__":
    yaml_path = r"./output/folder/homography.yaml"
    image_path = r"./data/RotatedChessBoard_renamed/40.png"
    point = (606.91235, 351.53717)

    H = load_homography_yaml(yaml_path)
    Pmapping, vis_image = back_project_point(H, image_path, point, show=False)
    print(f"投影图上点 {point} 反投影到原图坐标: {Pmapping}")

    os.makedirs("./output", exist_ok=True)
    save_path = "./output/40_back_projection.jpg"
    cv2.imwrite(save_path, vis_image)
    print(f"已保存: {save_path}")
