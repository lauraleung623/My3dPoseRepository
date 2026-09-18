import os
import cv2
import numpy as np


def rotate_image(image_path, folder_path, N=10, target_w=0, target_h=0,rotation_center=None):
    """
    读取 image_path 的图片，按 360/N 度为步长旋转，共生成 N 张图，
    保存到 folder_path 下，文件名为 int(360/N * i).png
    """
    os.makedirs(folder_path, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    img_center = (w / 2.0, h / 2.0)
    # 自定义旋转中心，如果不设置时图片中心
    if rotation_center is None:
        center = img_center
    else:
        center = rotation_center

    step = 360 / N
    for i in range(N):
        angle = step * i
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        if target_w > 0 and target_h > 0:
            new_w = target_w
            new_h = target_h
        else:
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
        # 填充按原图几何中心对齐到新画布中心，旋转中心只随这次平移移动
        M[0, 2] += (new_w / 2) - img_center[0]
        M[1, 2] += (new_h / 2) - img_center[1]
        # 原图旋转中心经完整仿射变换（旋转 + 画布填充平移）后的坐标
        rotation_center_new = (
            M[0, 0] * center[0] + M[0, 1] * center[1] + M[0, 2],
            M[1, 0] * center[0] + M[1, 1] * center[1] + M[1, 2],
        )
        rotated = cv2.warpAffine(img, M, (new_w, new_h),
                    borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        save_path = os.path.join(folder_path, f"{int(angle):03d}.png")
        cv2.imwrite(save_path, rotated)
        print(f"Saved: {save_path},image width={new_w},image height={new_h}")
    return rotation_center_new


def draw_center_on_images(rotation_center_new, radius, folder_path):
    """
    读取 folder_path 下所有图片，在 rotation_center_new 处画十字，
    并以该点为圆心、radius 为半径画圆，覆盖保存原图。
    """
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
    img_list.sort()
    if not img_list:
        print(f"未在文件夹中找到图片: {folder_path}")
        return

    cx, cy = int(rotation_center_new[0]), int(rotation_center_new[1])
    center = (cx, cy)
    color = (0, 0, 255)
    cross_size = 80
    thickness = 3

    for img_name in img_list:
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片: {img_path}")
            continue
        cv2.line(image, (cx - cross_size, cy), (cx + cross_size, cy), color, thickness)
        cv2.line(image, (cx, cy - cross_size), (cx, cy + cross_size), color, thickness)
        cv2.circle(image, center, int(radius), color, thickness)
        cv2.imwrite(img_path, image)
        print(f"Annotated: {img_path}")


if __name__ == "__main__":
    image_path = r"D:\AutoCalibration_code\Pic\chessboard.jpeg"
    folder_path = r".\outputImg"
    N=36
    target_w=3000
    target_h=3000
    rotation_center=(1000,800)  # 原图的坐标，不是新图的坐标
    #rotation_center_new=rotate_image(image_path, folder_path, N, target_w, target_h)
    rotation_center_new=rotate_image(image_path, folder_path, N, target_w, target_h,rotation_center)
    
    # 用于肉眼观察，中心是否求解正确
    draw_center_on_images(rotation_center_new, 1000, folder_path)

    print(f"Rotation center new=({rotation_center_new[0]:.2f},{rotation_center_new[1]:.2f})")
