
import pandas as pd
import numpy as np
import argparse
import cv2
import os
import sys


def gd_gu_frame_extractor(video_path,old_gt_csv_path,pose_csv_path):

    frame_number = 0

    cam = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)[:-4]
    
    output_path=os.path.join(os.getcwd())
    os.makedirs(output_path,exist_ok=True)
    
    old_strides_df = pd.read_csv(old_gt_csv_path)
    pose_csv_df=pd.read_csv(pose_csv_path)
    
    gdgu_df=old_strides_df[['GD','GU']].values.reshape(1,-1)
    
    gd_gu_list=[]
    for i in gdgu_df[0]:
        gd_gu_list.append(i)

    all_image_list=[]
    hstack_list=[]


    while cam.isOpened():

        ret_val, image = cam.read()

        if ret_val is False:
            break

        frame_number += 1
        if frame_number in gd_gu_list:
            df=pose_csv_df.loc[pose_csv_df['Frame']==frame_number]
            xmin=int(df['TL_X'].values)
            if xmin>=20:
                xmin=xmin-20
            xmax=int(df['BR_X'].values)+20
            ymin=int(df['TL_Y'].values)
            if ymin>=20:
                ymin=ymin-20
            ymax=int(df['BR_Y'].values)+20
            im=image[ymin:ymax,xmin:xmax]
            all_image_list.append(im)
            
    for i in range(0,len(all_image_list),2):
        i1=all_image_list[i]
        i2=all_image_list[i+1]
        i1=cv2.resize(i1,(500,500))
        i2=cv2.resize(i2,(500,500))
        h=np.hstack([i1,i2])
        hstack_list.append(h)
    vstack=np.vstack(hstack_list)

    path=os.path.join(output_path,f"collage_{video_name}.jpg")
    cv2.imwrite(path,vstack)
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--video_path", type=str, help="Enter path to video", required=True
    )
    
    parser.add_argument("--old_csv_path", type=str, help="Path to 3GT CSV")
    parser.add_argument("--pose_csv_path", type=str, help="Path to id CSV")
    opt = parser.parse_args()

    
    video_path = opt.video_path
    old_csv_path = opt.old_csv_path
    pose_csv_path = opt.pose_csv_path
        
    print("Saving GD GU Frames :")

    gd_gu_frame_extractor(video_path,old_csv_path,pose_csv_path)