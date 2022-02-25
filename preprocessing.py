import cv2
import mediapipe as mp
import numpy as np
import dlib
import time
import csv
from pathlib import Path

route = f"{Path(__file__).parent}/"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

found = 0
BG_COLOR = ( 255,255,255 ) 
Threshold = 0.8

def halfp(arr1,arr2):
    x = int((arr1[0]+arr2[0])/2)
    y = int((arr1[1]+arr2[1])/2)
    
    return [x,y]

#return nobg_img / pose landmarks
def bgremoval(route):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True, 
        smooth_segmentation = True,
        min_detection_confidence=0.9  ) as pose:
        
        pose_landmarks = []
        temp = []
        pose_border = []
    
        image = cv2.imread(route)
        image_height, image_width, _ = image.shape
        
        maxy = image_width
        miny = 0
        
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > Threshold
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        
        gray = cv2.cvtColor(annotated_image,cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        for a in range(len(contours)):
            for b in range(len(contours[a])):
                miny = max(miny,contours[a][b][0][1])
                maxy = min(maxy,contours[a][b][0][1])
        
        length = miny - maxy

        if results.pose_landmarks:   
            npoints = []
            hipy = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)/2
            shouldery = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2 
            mouthy = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y)/2 
            
            necky = int((shouldery+mouthy)/2*image_height)
            
            kneey = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)/2 
            ankley = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)/2 
            
            
            pose_border.append(necky)
            
            pose_border.append(int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)/2 * image_height))
            
            pose_border.append(int((((hipy+shouldery)/2)+hipy)/2*image_height))
            
            pose_border.append(int((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)/2 * image_height))
            
            pose_border.append(int((hipy+kneey)/2*image_height))
            
            pose_border.append(int((ankley+kneey)/2*image_height))
            
            for p in range(33):
            
                temp.append([int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(p).value].x * image_width),
                                      int(results.pose_landmarks.landmark[mp_pose.PoseLandmark(p).value].y * image_height)])
                
            npoints.append(halfp(temp[11],temp[12]))
            npoints.append(halfp(temp[12],temp[14]))
            npoints.append(halfp(temp[16],temp[14]))
            npoints.append(halfp(temp[11],temp[13]))
            npoints.append(halfp(temp[13],temp[15]))
            npoints.append(halfp(temp[24],temp[26]))
            npoints.append(halfp(temp[26],temp[28]))
            npoints.append(halfp(temp[23],temp[25]))
            npoints.append(halfp(temp[27],temp[25]))
            
            npoints.append([halfp(npoints[0],temp[12])[0],halfp(npoints[0],halfp(temp[23],temp[24]))[1]])
            npoints.append([halfp(npoints[0],temp[11])[0],halfp(npoints[0],halfp(temp[23],temp[24]))[1]])
            
            npoints.append(halfp(halfp(npoints[9],npoints[10]),npoints[0]))
            npoints.append(halfp(halfp(npoints[9],npoints[10]),halfp(temp[23],temp[24])))
            
            npoints.append(halfp(npoints[5],temp[24]))
            npoints.append(halfp(npoints[5],temp[26]))
            npoints.append(halfp(npoints[6],temp[26]))
            npoints.append(halfp(npoints[6],temp[28]))
            
            npoints.append(halfp(npoints[7],temp[23]))
            npoints.append(halfp(npoints[7],temp[25]))
            npoints.append(halfp(npoints[8],temp[25]))
            npoints.append(halfp(npoints[8],temp[27]))


            for x in range(len(temp)):
                pose_landmarks.append(temp[x])
                
            for x in npoints:
                pose_landmarks.append(x)
            
            for a in range(6):
                pose_border[a] = round(pose_border[a] / length,4)
            


    return pose_border,pose_landmarks,annotated_image
