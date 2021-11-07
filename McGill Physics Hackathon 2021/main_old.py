'''
Created on Nov. 6, 2021

@author: kuba
'''
import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import time
from scipy.spatial.transform import Rotation as R
from tkinter.tix import Balloon

start_time = time.clock_gettime(0)
curr_time = start_time

camera_calib_path = os.getcwd()+"/calib_data/"
mtx = np.load(camera_calib_path+"camera_mtx"+".npy")
dist =np.load(camera_calib_path+"camera_dist"+".npy")
# Start streaming


# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
profile = pipe.start(cfg)

align_to = rs.stream.color
align = rs.align(align_to)

class ball:
    def __init__(self,x,y,size,e):
        self.x = x 
        self.y = y 
        self.e = e 
        self.vx = 0.0
        self.vy = 0.0
        self.pos = np.array([x,y])
        self.vel = np.array([0.0,0.0])
        self.size = size
    def draw(self,img):
        img = cv2.circle(img,np.asanyarray(self.pos,np.int),self.size,(0,255,255),-1)
        return img
        
def update_pos(obj):
    t = 1/30
    obj.vel[1] += 9.81*t
    obj.pos += obj.vel

sph = ball(500.0,500.0,50,1)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        
        color_image = np.array(np.fliplr(color_image))
        
        update_pos(sph)
        color_image = sph.draw(color_image)
        
        
        depth_image = np.fliplr(depth_image)
        
        depth_image = depth_image/1000
        hand = cv2.inRange(depth_image, (0.25), (0.5))
        #hand = (depth_image < 0.5) & (depth_image != 0) & (depth_image > 0.25)

        #perform the morphological opening and closing
        kernel = np.ones((4,4),np.uint8)
        
        #hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, kernel)
        #hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel)
        hand = cv2.morphologyEx(hand, cv2.MORPH_DILATE, kernel)
        
        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
        
        # Show images
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Depth', hand.astype(np.uint8)*255)
        cv2.imshow('Depth', hand)
        cv2.waitKey(1)
finally:
    # Stop streaming
    pipe.stop()