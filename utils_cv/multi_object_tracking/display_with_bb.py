""" Takes in string labels as well
Adapted from https://github.com/ZQPei/deep_sort_pytorch.
"""
import cv2
import pandas as pd
import argparse
import os
from PIL import Image

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_all_labels(id_list):
    """
    :param id_list:
    :return:
    """
    color_list = []
    id_list2 = list(range(len(id_list)))                  
    for i in id_list2:
        color = [int((p * ((i+1) ** 5 - i + 1)) % 255) for p in palette] #[int((p * ((label) ** 2 - label + 1)) % 255) for p in palette]
        color_list.append(tuple(color))
        
    id_color_dict=dict(zip(id_list, color_list ))
       
    return(id_color_dict)

def draw_boxes(image, cur_frame, id_color_dict, offset=(0, 0), id_present=True):
    """

    :param image:
    :param cur_frame:
    :param offset:
    :return:
    """
   
    cur_ids = cur_frame["id"].tolist()  
    
    for i,id in enumerate(cur_ids): 
        bbox_left = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_left"].values[0]))
        bbox_top = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_top"].values[0]))
        bbox_right = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_right"].values[0]))
        bbox_bottom = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_bottom"].values[0]))

        bbox_left += offset[0]
        bbox_right += offset[0]
        bbox_top += offset[1]
        bbox_bottom += offset[1]
        
        # Box text and bar       
        color = id_color_dict.get(id)       
        label = str(cur_frame.loc[cur_frame["id"] == id, "label"].values[0])
     
        # Last two args of getTextSize() are font_scale and thickness
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        cv2.rectangle(image, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, 3)
        # cv2.rectangle(image, (bbox_left, bbox_top), (bbox_left+t_size[0]+3, bbox_top+t_size[1]+4), color, -1)
        if id_present: 
            cv2.putText(image, "id_"+label, (bbox_left, bbox_top+t_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) #[255,255,255],1)
     
    return image


def process_video(video_path, labels_path, save_path, fps, labels_format, img_path, id_present=True):
    """
    :param video_path:
    :param labels_path:
    :param save_path:
    :param fps:
    :param labels_format:
    :return:
    """
    video = cv2.VideoCapture()
    video.open(video_path)
    #video.set(cv2.CAP_PROP_FPS , fps)# KIP

    image_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (image_width, image_height))

    print("video_path:", video_path)
    print("video fps:", video.get(cv2.CAP_PROP_FPS))
    
    if labels_format == "mot_challenge": #xywh, with video data
        df = pd.read_csv(labels_path, header=None)
        df = df.iloc[:,:6]
        df.columns = ["frame", "id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"]
        df["bbox_right"] = df["bbox_left"] + df["bbox_width"]
        df["bbox_bottom"] = df["bbox_top"] + df["bbox_height"]
        df["label"] = df["id"]
        
        #Assign color per id:
        id_color_dict = compute_color_for_all_labels(df["id"].unique().tolist())
        
        frame_idx = 1
        while video.grab():
            _, cur_image = video.retrieve()
            cur_frame = df[df["frame"] == frame_idx]               
            if not cur_frame.empty:               
                cur_image = draw_boxes(cur_image, cur_frame, id_color_dict)
            writer.write(cur_image)
            frame_idx += 1

    elif labels_format == "pascal_voc": # xmin, ymin, xmax, ymax, with image seq data
        df = pd.read_csv(labels_path)
        df.columns = ["image", "xmin", "ymin", "xmax", "ymax", "label"]
        df["id"]=df["label"]      
        #Assign color per id:
        id_color_dict = compute_color_for_all_labels(df["id"].unique().tolist())
       
        #Convert to mot_challenge label format: 
        df["bbox_left"]=df["xmin"]
        df["bbox_top"]=df["ymin"]
        df["bbox_right"] = df["xmax"]
        df["bbox_bottom"] = df["ymax"]
        
        #wrangle image column to assign frame number
        df["time"] = df["image"].apply(lambda x: float(x.split("=")[1].replace(".jpg", "")))
        df.sort_values(by="time", inplace=True)
        img_list = df["image"].unique().tolist()
        key_list = list(range(1,len(img_list)+1))
        framemap_dict=dict(zip(img_list, key_list))
        df["frame"] = df["image"].apply(lambda x: framemap_dict.get(x)) 
       
        for im_name in  df["image"].unique().tolist():  
            cur_frame = df[df["image"]==im_name]       
            cur_image = draw_boxes(cv2.imread(os.path.join(img_path, im_name)),cur_frame, id_color_dict, id_present=id_present) 
            writer.write(cur_image)

            cv2.waitKey(0)
                    
    else:
        raise Exception("Format of input labels not supported.")

def process_images(img_path, labels_path, save_path, labels_format, image_frame_dict):
    """
    :param img_path:
    :param labels_path:
    :param save_path:
    :param labels_format:
    :return:
    """
    if labels_format == "mot_challenge":
        df = pd.read_csv(labels_path, header=None)        
        df = df.iloc[:,:6]
        df.columns = ["frame", "id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"]
        df["bbox_right"] = df["bbox_left"] + df["bbox_width"]
        df["bbox_bottom"] = df["bbox_top"] + df["bbox_height"]
        df["label"] = df["id"]

        #Assign color per id:
        id_color_dict = compute_color_for_all_labels(df["id"].unique().tolist())

        #Add bboxes and id-labels in each image in the sequence        
        for im_name in image_frame_dict.keys():
            cur_frame = df[df["frame"]==image_frame_dict.get(im_name)]  
            img1= cv2.imread(os.path.join(img_path, im_name))                       
            cur_image = draw_boxes(cv2.imread(os.path.join(img_path, im_name)),cur_frame, id_color_dict, id_present=True)  
            cv2.imwrite(os.path.join(save_path, im_name), cur_image)
            
    print("Saving images with bboxes and labels to:", save_path)
    
    return()
    

def main():
    default_video_path = "../ground_truth/Famalicao_Half1_Trim_00_45-1_07.mp4" #TODO: remove Benfica filenames
    default_labels_path = "../ground_truth/4_20SecsFamalicao_Final_Annotations.csv"
    default_save_path = "../ground_truth/ground_truth_annotations_30fps.mp4"
    default_fps = 30
    default_labels_format = "mot_challenge"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=default_video_path)
    parser.add_argument("--labels_path", type=str, default=default_labels_path)
    parser.add_argument("--save_path", type=str, default=default_save_path)
    parser.add_argument("--fps", type=int, default=default_fps)
    parser.add_argument("--labels_format", type=str, default=default_labels_format)
    parser.add_argument("--img_path", type=str, default=default_img_path)
    #parser.add_argument("--id_present", type=str_to_bool, nargs='?', const=True, default=True) #TODO: check if boolean arg works
    args = parser.parse_args()

    process_video(args.video_path, args.labels_path, args.save_path, args.fps, args.labels_format, args.img_path)


if __name__ == '__main__':
    main()
