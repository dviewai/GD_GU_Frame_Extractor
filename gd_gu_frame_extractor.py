
import pandas as pd
import numpy as np
import argparse
import cv2
import os
import sys


def gd_gu_frame_extractor(video_path,old_gt_csv_path):

    frame_number = 0

    cam = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)[:-4]
    
    output_path=os.path.join(os.getcwd(),video_name)
    os.makedirs(output_path,exist_ok=True)
    
    old_strides_df = pd.read_csv(old_gt_csv_path)
    
    
    gdgu_df=old_strides_df[['GD','GU']].values.reshape(1,-1)
    
    gd_gu_list=[]
    for i in gdgu_df[0]:
        gd_gu_list.append(i)


    while cam.isOpened():

        ret_val, image = cam.read()

        if ret_val is False:
            break

        frame_number += 1
        if frame_number in gd_gu_list:

            path=os.path.join(output_path,f"{frame_number}.jpg")
            cv2.imwrite(path,image)
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--video_path", type=str, help="Enter path to video", required=True
    )
    
    parser.add_argument("--old_csv_path", type=str, help="Path to 3GT CSV")
    opt = parser.parse_args()

    
    video_path = opt.video_path
    old_csv_path = opt.old_csv_path
        
    print("Saving GD GU Frames :")

    gd_gu_frame_extractor(video_path,old_csv_path)