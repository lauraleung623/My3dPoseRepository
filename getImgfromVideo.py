import os
import shutil
import cv2


def get_video_frame_count(video_path):
    """
    读取 video_path 视频，返回并打印视频的总帧数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    cap.release()
    return total_frames


def get_img_from_video(video_path):
    """
    读取 video_path 视频，将每一帧保存为图片。
    图片保存在与视频同名的文件夹下（与视频同目录），
    文件名以帧序号命名，补全为3位数，如 000.png, 001.png ...
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 创建与视频同名的文件夹（与视频同目录）
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(video_dir, video_name)
    os.makedirs(output_folder, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        save_path = os.path.join(output_folder, f"{frame_idx:03d}.png")
        cv2.imwrite(save_path, frame)
        frame_idx += 1

    cap.release()
    print(f"共保存 {frame_idx} 帧图片到: {output_folder}")
    return output_folder

def rename_and_copy_imgs(img_folder, target_folder):
    """
    从 img_folder 读取所有图片路径，按名称排序后，
    从 00 开始重新命名（补全为2位数），并复制到 target_folder。
    """
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(img_exts)]
    img_list.sort()

    if not img_list:
        print(f"未在文件夹中找到图片: {img_folder}")
        return None

    if len(img_list) > 99:
        print("图片过多")
        return None

    os.makedirs(target_folder, exist_ok=True)

    for idx, img_name in enumerate(img_list):
        src_path = os.path.join(img_folder, img_name)
        ext = os.path.splitext(img_name)[1]
        dst_name = f"{idx:02d}{ext}"
        dst_path = os.path.join(target_folder, dst_name)
        shutil.copy(src_path, dst_path)

    print(f"共复制 {len(img_list)} 张图片到: {target_folder}")
    return target_folder


if __name__ == "__main__":
    video_path = r"./data/RotatedChessBoard.mp4"
    # 已提取过了
    #get_video_frame_count(video_path)
    #get_img_from_video(video_path)

    # 重命名图片
    img_folder = r"./data/RotatedChessBoard_SelectedWrongIndex"
    target_folder = r"./data/RotatedChessBoard_renamed"
    rename_and_copy_imgs(img_folder, target_folder)
