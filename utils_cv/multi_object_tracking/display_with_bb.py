import cv2
import pandas as pd

def convert_trackingbboxes_txt(prediction_dict, result_file_path) -> None: # called by evaluate() in TrackingLearner class
    """ Takes in output of predict function. Outputs resulting tracking result as txt file, readable by "motmetrics" package.
        Args:
            prediction_dict (Dict[int, List[TrackingBbox]]}): output of predict function, i.e. dictionary of trackingbboxes objects with frame number as key
            result_file_path: path at which text file is output, in format [frame_id][track_id][left][top][width][height]    
    """
    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    with open(result_file_path, 'w') as f:    
        for key_i in sorted(prediction_dict.keys()): #each frame_id
            for bbox_i in prediction_dict.get(key_i): #each bbox
                print('%d,%d,%.5f,%.5f,%.5f,%.5f,-1,-1,-1' %(bbox_i.frame_id, bbox_i.track_id, bbox_i.left, bbox_i.top, bbox_i.right-bbox_i.left, bbox_i.bottom-bbox_i.top) , file=f)     

                
def convert_trackingbboxes_video(prediction_dict, video_path, frame_rate, result_video_path="./tracked_video.mp4")-> None : 
    """ Takes in output of predict function. Outputs resulting tracking video.
        Args:
            prediction_dict (Dict[int, List[TrackingBbox]]}): output of predict function, i.e. dictionary of trackingbboxes objects with frame number as key
            video_path (str): path of the raw video over which to overlay tracking bboxes results.
            frame_rate (int): frame rate of video
            result_video_path (str): path at which video with tracked results (bboxes and id overlayed) is output

        Returns (str): 
            path at which the txt file containing the bboxes and ids in MOT format has been saved.    
    """    
    # Convert prediction_dict to dataframe in MOT challenge format    
    list_bboxes = []
    for key_i in sorted(prediction_dict.keys()): #each frame_id
        for bbox_i in prediction_dict.get(key_i): #each bbox
            list_bboxes.append([bbox_i.frame_id, bbox_i.track_id, bbox_i.left, bbox_i.top, bbox_i.right-bbox_i.left, bbox_i.bottom-bbox_i.top])
    df_bboxes = pd.DataFrame(list_bboxes, columns= ["frame", "id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"])
    
    # Call functions to create video
    process_df_video(df_bboxes, video_path, frame_rate, result_video_path)


def process_df_video(df_bboxes, video_path, frame_rate, result_video_path)-> None: #(video_path, labels_path, save_path, fps, labels_format, img_path, id_present=True):
    """ Takes in a df of tracking results and the input dataset video, and outputs the resulting video with the bboxes,ids overlayed
        Args: 
            df_bboxes (Pandas.DataFrame): df of tracking results with columns ["frame", "id", "bbox_left", "bbox_top", "bbox_width", "bbox_height"]
            video_path (str): path of raw video 
            frame_rate (int): frame rate at which to write the new video       
            result_video_path (str): path of tracking video to be output   
    """
    
    # Read video and initialize new tracking video
    video = cv2.VideoCapture()
    video.open(video_path)
    
    image_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(result_video_path, fourcc, frame_rate, (image_width, image_height))
    
    #Assign bbox color per id:
    id_color_dict = compute_color_for_all_labels(df_bboxes["id"].unique().tolist())
    
    # Create images and add to video writer, adapted from https://github.com/ZQPei/deep_sort_pytorch  
    frame_idx = 1
    while video.grab():
        _, cur_image = video.retrieve()
        cur_frame = df_bboxes[df_bboxes["frame"] == frame_idx]               
        if not cur_frame.empty:               
            cur_image = draw_boxes(cur_image, cur_frame, id_color_dict)
        writer.write(cur_image)
        frame_idx += 1

def draw_boxes(image, cur_frame, id_color_dict):
    """ Function to overlay bbox and id labels on current image frame
    Args:
        image (Numpy.ndarray): array of raw image at specified frame
        cur_frame (int): frame number of current image to extract id label
        id_color_dict (dict[int, (int,int,int)]): color dictionary to extract bbox color specific to id        
    Returns:
        image (Numpy.ndarray): array of raw image at specified frame, with overlaid bboxes+id labels
    """
   
    cur_ids = cur_frame["id"].tolist()  
    
    for i,id in enumerate(cur_ids): 
        bbox_left = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_left"].values[0]))
        bbox_top = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_top"].values[0]))
        bbox_right = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_right"].values[0]))
        bbox_bottom = int(round(cur_frame.loc[cur_frame["id"] == id, "bbox_bottom"].values[0]))
    
        # Box text and bar       
        color = id_color_dict.get(id)       
        label = str(cur_frame.loc[cur_frame["id"] == id, "label"].values[0])
     
        # Last two args of getTextSize() are font_scale and thickness
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        cv2.rectangle(image, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, 3)
        # cv2.rectangle(image, (bbox_left, bbox_top), (bbox_left+t_size[0]+3, bbox_top+t_size[1]+4), color, -1)
        cv2.putText(image, "id_"+label, (bbox_left, bbox_top+t_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3) #[255,255,255],1)
     
    return image

        
def compute_color_for_all_labels(id_list):
    """ Takes in a list of unique ids, to produce corresponding unique color palettes
    Args:
        id_list (list[int]):list of integers for unique id numbers
    Returns:
        id_color_dict (dict[int, (int,int,int)])
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    
    color_list = []
    id_list2 = list(range(len(id_list)))   

    #Adapted from https://github.com/ZQPei/deep_sort_pytorch    
    for i in id_list2:
        color = [int((p * ((i+1) ** 5 - i + 1)) % 255) for p in palette] #[int((p * ((label) ** 2 - label + 1)) % 255) for p in palette]
        color_list.append(tuple(color))
        
    id_color_dict=dict(zip(id_list, color_list ))
       
    return(id_color_dict)



