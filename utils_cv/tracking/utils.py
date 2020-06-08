import pandas as pd
import os
import shutil
from PIL import Image


def convert_vott_MOTxywh(vott_path, converted_path) -> None:
    """ Utility function to convert VOTT dataset (.jpgs and .csv files) to MOT Challenge format (ground truth data in xywh) to be able to be read by FairMOT wrapper
        Args: 
            vott_path, e.g.  "./data/carcans_vott-csv-export/"
            converted_path, .e.g.  "./data/carcans_MOTformat/"
    """
    df = pd.read_csv(
        os.path.join(
            vott_path,
            [f for f in os.listdir(vott_path) if f.endswith(".csv")][0],
        )
    )

    # Sort image column and order by frame number
    df["time"] = df["image"].apply(
        lambda x: x.split("=")[1].replace(".jpg", "")
    )
    # file_list = sorted(os.listdir(im_path), key = lambda x:int(x.split(".")[0].replace("frame","")))
    list1 = sorted(list(dict.fromkeys(df["time"])), key=lambda x: float(x))
    dict1 = dict(zip(list1, list(range(1, len(list1) + 1))))
    df["frame"] = df["time"].apply(lambda x: dict1.get(x))
    df.sort_values(by="frame", inplace=True)

    # Add dict for id matching from 1, 2...
    list_label = list(set(df["label"]))
    dict_label = dict(zip(list_label, range(1, len(list_label) + 1)))
    df["label_numbered"] = df["label"].apply(lambda x: dict_label.get(x))

    # Write to .txt file
    path1 = os.path.join(vott_path, "single_vid", "gt")
    if not os.path.exists(path1):
        os.makedirs(path1)

    file = os.path.join(path1, "gt.txt")
    with open(file, "w") as f:

        for (
            i,
            row,
        ) in (
            df.iterrows()
        ):  # in format [frame number] [id number] [bbox left] [bbox top] [bbox width] [bbox height][confidence score][class][visibility]
            print(
                "%d,%d,%.5f,%.5f,%.5f,%.5f,-1,-1,-1"
                % (
                    row["frame"],
                    row["label_numbered"],
                    row["xmin"],
                    row["ymin"],
                    row["xmax"] - row["xmin"],
                    row["ymax"] - row["ymin"],
                ),
                file=f,
            )

    # Create image folder
    dest_path = os.path.join(vott_path, "single_vid", "img1")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for file_i in os.listdir(vott_path):
        if file_i.split(".")[-1] in ["jpg", "jpeg", "png", "tif"]:
            frame_no = dict1.get(file_i.split("=")[1].replace(".jpg", ""))
            img_filename = "{:05.0f}".format(frame_no) + ".jpg"
            shutil.copy(
                os.path.join(vott_path, file_i),
                os.path.join(dest_path, img_filename),
            )

    # Create seqinfo folder
    frame_rate = 30
    img_filename = os.listdir(os.path.join(vott_path, "single_vid", "img1"))[0]
    seq_length = len(os.listdir(os.path.join(vott_path, "single_vid", "img1")))
    im_width, im_height = im = Image.open(
        os.path.join(vott_path, "single_vid", "img1", img_filename)
    ).size
    im_ext = "." + img_filename.split(".")[-1]

    file = os.path.join(vott_path, "single_vid", "seqinfo.ini")
    with open(file, "w") as f:
        print("[Sequence]", file=f)
        print("name=single_vid", file=f)
        print("imDir=img1", file=f)
        print("frameRate=%d" % frame_rate, file=f)
        print("seqLength=%d" % seq_length, file=f)
        print("imWidth=%s" % str(im_width), file=f)
        print("imHeight=%s" % str(im_height), file=f)
        print("imExt=%s" % im_ext, file=f)
