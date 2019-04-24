from utils_cv.classification.data import imagenet_labels, downsize_imagelist


def test_imagenet_labels():
    # Compare first five labels for quick check
    IMAGENET_LABELS_FIRST_FIVE = (
        "tench",
        "goldfish",
        "great_white_shark",
        "tiger_shark",
        "hammerhead",
    )

    labels = imagenet_labels()
    for i in range(5):
        assert labels[i] == IMAGENET_LABELS_FIRST_FIVE[i]


def test_downsize_imagelist(tiny_ic_data_path, tmp):
    im_list = ImageList.from_folder(tiny_ic_data_path)
    max_dim = 50
    downsize_imagelist(im_list, tmp, max_dim)
    im_list2 = ImageList.from_folder(tmp)
    assert len(im_list) == len(im_list2)
    for im_path in im_list2.items:
        assert min(Image.open(im_path).size) <= max_dim
