import cv2
import os
import glob
import argparse


def vid_to_seq(seq_dir, vid_path, fps):
    """

    :param seq_dir:
    :param vid_path:
    :param fps:
    :return:
    """
    vdo = cv2.VideoCapture()
    vdo.open(vid_path)
    
    if not os.path.exists(seq_dir):
        os.mkdir(seq_dir)

    idx_frame = 0
    while vdo.grab():
        _, image = vdo.retrieve()
        cv2.imwrite(os.path.join(seq_dir, "frame{:06d}.jpg".format(idx_frame)), image)
        idx_frame += 1


def seq_to_vid(seq_dir, vid_path, fps):
    """

    :param seq_dir:
    :param vid_path:
    :param fps:
    :return:
    """
    img_array = []
    files = glob.glob(os.path.join(seq_dir, '*.gif'))
    files.extend(glob.glob(os.path.join(seq_dir, '*.png')))
    files.extend(glob.glob(os.path.join(seq_dir, '*.jpg')))
    
    print("files:", files)
    files = sorted(files)#, key = lambda x:int(x.split(".")[0]))
    print("files sorted:", files)
    for file in files:
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    for image in img_array:
        out.write(image)
    out.release()


def main():
    default_seq_dir = "../ground_truth/seq_annotations_30fps"
    default_vid_path = "../ground_truth/ground_truth_annotations_30fps.mp4"
    default_run_option = "vid_to_seq"
    default_fps = 30

    # default_seq_dir = "../ground_truth/seq_benfica_25fps"
    # default_vid_path = "../ground_truth/ground_truth_benfica_25fps.mp4"
    # default_run_option = "vid_to_seq"
    # default_fps = 25

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", type=str, default=default_seq_dir)
    parser.add_argument("--vid_path", type=str, default=default_vid_path)
    parser.add_argument("--run_option", type=str, default=default_run_option)
    parser.add_argument("--fps", type=int, default=default_fps)
    args = parser.parse_args()

    if args.run_option == "vid_to_seq":
        vid_to_seq(args.seq_dir, args.vid_path, args.fps)
    elif args.run_option == "seq_to_vid":
        seq_to_vid(args.seq_dir, args.vid_path, args.fps)
    else:
        raise Exception("Run option not supported.")


if __name__ == '__main__':
    main()
