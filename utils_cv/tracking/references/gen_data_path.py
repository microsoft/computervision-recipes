import os
import glob
import _init_paths


def gen_caltech_path(root_path):
    label_path = 'Caltech/data/labels_with_ids'
    real_path = os.path.join(root_path, label_path)
    image_path = real_path.replace('labels_with_ids', 'images')
    images_exist = sorted(glob.glob(image_path + '/*.png'))
    with open('../src/data/caltech.all', 'w') as f:
        labels = sorted(glob.glob(real_path + '/*.txt'))
        for label in labels:
            image = label.replace('labels_with_ids', 'images').replace('.txt', '.png')
            if image in images_exist:
                print(image[22:], file=f)
    f.close()


if __name__ == '__main__':
    root = '/data/yfzhang/MOT/JDE'
    gen_caltech_path(root)