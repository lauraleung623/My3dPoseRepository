import os
import cv2

def generate_video(img_folder, file_name, fps=10):
    # 只保留常见图片格式的文件
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(img_exts)]
    img_list.sort()

    if not img_list:
        print("未在文件夹中找到图片: {}".format(img_folder))
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(file_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取第一张图片，获取宽高，用于初始化VideoWriter
    first_img_path = os.path.join(img_folder, img_list[0])
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        print("无法读取图片: {}".format(first_img_path))
        return
    height, width = first_frame.shape[:2]

    # mp4格式编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

    for img_name in img_list:
        img_path = os.path.join(img_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print("跳过无法读取的图片: {}".format(img_path))
            continue
        # 如果尺寸不一致，resize到统一大小
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)

    video_writer.release()
    print("视频已保存到: {}".format(file_name))


if __name__ == "__main__":
    img_folder=r'D:/AutoCalibration_code/Tool/Rotation_Img/Center1367_1556_wcircle'
    output_path=r'D:/AutoCalibrate/Pic/output_video'
    file_name='Sep16.mp4'
    file_name=os.path.join(output_path,file_name)
    fps=10  # 可根据需要修改帧率

    generate_video(img_folder,file_name,fps)
    print("Completed!")
