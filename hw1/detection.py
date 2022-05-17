import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    positions = list() # 宣告一個list用來裝detectData.txt
    with open(dataPath) as file: # 打開儲存停車格座標的 txt 檔案
      N = int(file.readline())  # 輸入座標數量
      for pos in range(N): 
        line = file.readline().split() # 輸入停車格座標並以空格切割成 list
        positions.append( tuple(map(int,line)) ) # 轉成 tuple 之後丟進 position 這個 list 裡面

    cap = cv2.VideoCapture(os.path.join(dataPath,"..","video.gif")) # 讀取 gif 檔案
    cur_frame = 0 # 記錄迴圈跑到第幾幀
    gifs = [] # 儲存處理過的每一幀（方便輸出影片）

    while True:
      cur_frame += 1
      detects = []
      _, img = cap.read() # 讀取一幀
      if img is None: #如果讀取完所有的圖片就停止迴圈
        break

      for pos in positions:
        cropped = crop(*pos , img) # 將座標傳進crop函數當中裁切
        cropped = cv2.resize(cropped , (36 , 16)) # resize 圖片
        cropped = cv2.cvtColor(cropped , cv2.COLOR_RGB2GRAY) # 轉成灰階
        detects.append(clf.classify(cropped)) # 將圖片丟進classifty function分類

      for i in range(len(detects)):
        # 如果 classify 是 True ，就在該車格上畫出矩形
        if detects[i]: 
          pos = [[positions[i][j] , positions[i][j+1]] for j in range(0,8,2)]
          pos[2],pos[3] = pos[3],pos[2] # 交換座標2跟3的順序，線條才會連起來
          pos = np.array(pos , np.intc) # 轉成 numpy array
          cv2.polylines(img , [pos] , color=(0,255,0) , isClosed=True) # 用綠色畫出停車格

      if cur_frame == 1:
        cv2.imwrite(f"first_frame_of_adaboost.png" , img) # 輸出第一幀

    # End your code (Part 4)


# detect('data/detect/detectData.txt', clf)