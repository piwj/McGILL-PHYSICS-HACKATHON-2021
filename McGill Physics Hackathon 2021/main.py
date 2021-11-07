'''
Created on Nov. 7, 2021

@author: Saakshi Turakhia
@author: Jakub Piwowarczyk
'''

import numpy as np
import pyrealsense2 as rs
import cv2
import time

class ball:
    # define a ball object for use in the simulations
    
    def __init__(self, x, y, rad, e):
        # initialize the position, velocity, coefficient of restitution, and radius
        self.pos = np.array([x, y], np.float)
        self.vel = np.array([0.0, 0.0])
        self.e = float(e) 
        self.rad = rad
        self.is_collided = False

    def draw_ball(self, img):
        # draw the ball onto a color frame
        img = cv2.circle(img, np.asanyarray(self.pos,np.int), self.rad, (0,255,255), -1)
        return img
    
    def draw_ball_mask(self, img):
        # draw a binary mask with the position of the ball
        img = cv2.circle(img, np.asanyarray(self.pos,np.int), self.rad,(255), -1)
        return img
    
    def check_collision(self, hand):
        # create a mask for the ball to compare to the user input
        ball_mask = np.zeros_like(hand)
        ball_mask = self.draw_ball_mask(ball_mask)
        ball_mask = cv2.inRange(ball_mask, (254), (255))
        
        # calculate whether there is a collision, if there is calculate where and how significant it is
        and_arr = cv2.bitwise_and(ball_mask, hand)
        y, x = np.where(and_arr > 0)
        if (len(x) == 0) and (len(y) == 0):
            return np.array([]), 0
        else:
            return np.array([np.mean(x), np.mean(y)]), np.count_nonzero(and_arr)
        
    def simulate_contact(self, col_pos, col_amt):
        # calculate the angle of contact
        del_x = (self.pos[0] - col_pos[0])
        del_y = (self.pos[1] - col_pos[1])
        ang = np.arctan2(del_y,del_x)
    
        # if the ball velocity is in a similar path to the normal, simulate contact
        # utilize the coefficient of restitution in an approximated formula
        if np.abs(ang - self.vel_ang) > np.pi/2:
            self.vel[0] = self.e * self.vel[0]
            self.vel[1] = self.e * self.vel[1]
            
            velx = self.vel[0] * np.sin(ang) + self.vel[1] * np.cos(ang)
            vely = self.vel[1] * np.sin(ang) + self.vel[0] * np.cos(ang)
            
            self.vel[0] = velx
            self.vel[1] = vely
            
        # simulate a contact force
        force = .1
        self.vel[0] += col_amt * force * np.cos(ang)
        self.vel[1] += col_amt * force * np.sin(ang)
        
    def update_pos(self, delta_time):
        # process ball velocity and gravity
        t = delta_time
        
        # if the ball is colleded, then do not accelerate from gravity
        if not self.is_collided:
            self.vel[1] += 990.81 * t # this is in px/s^2
        
        # process velocities
        self.pos += self.vel * t
    
    @property
    def vel_ang(self):
        # calculate the direction the ball is going
        return np.arctan2(self.vel[1], self.vel[0]) 

def spawn_ball(event, x, y, flags, param):
    # add a new ball to the list of balls on click
    global balls

    if event == cv2.EVENT_LBUTTONDOWN:
        balls.append(ball(x, y, 60, 0.5))

def main(): 
    
    global balls
    
    # get current time
    curr_time = time.clock_gettime(0)
    
    # setup the stream from the realsense camera
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    
    # align color and depth frames
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # define and empty ball vector
    balls = []
    
    try:
        while True:
            # wait for a pair of depth and color frames
            frames = pipe.wait_for_frames()
            
            # align the color and depth frames
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                # wait for frames if there aren't any
                continue
            
            # calculate time difference for physics
            prev_curr_time = curr_time
            curr_time  = time.clock_gettime(0)
            delta_time = curr_time - prev_curr_time
    
            # get color frame and flip so it it more intuitive
            color_image = np.array(color_frame.get_data())
            color_image = np.array(np.fliplr(color_image))  
            
            # get the depth frame and convert to meters
            depth_image = np.fliplr(np.array(depth_frame.get_data()))
            depth_image = depth_image/1000
            
            # select only objects that are within 0.25 and 0.5 meters
            hand = cv2.inRange(depth_image, (0.25), (0.5))
    
            # perform a basic smoothing of our mask
            kernel = np.ones((4,4),np.uint8)      
            hand = cv2.morphologyEx(hand, cv2.MORPH_DILATE, kernel)
    
            # add a slight green tinge onto our color image to the input
            color_mask = np.zeros_like(color_image)
            color_mask[:] = (0,255,0)
            color_mask = cv2.bitwise_and(color_mask,color_mask,mask=hand)
            color_image = cv2.addWeighted(color_mask,0.3,color_image,1,0)
    
            # calculate physics for every spawned ball and draw on frame
            for obj in balls:   
                # check for collisions with the user input
                col_pos,col_amt = obj.check_collision(hand)
                
                # if collisions, process collision mechanics
                if col_pos.__len__() != 0:
                    obj.simulate_contact(col_pos,col_amt)
                    obj.is_collided = True
                else:
                    obj.is_collided = False

                # update position and draw
                obj.update_pos(delta_time)
                color_image = obj.draw_ball(color_image)
            
            # show frame and link callback for generating balls
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.setMouseCallback("RealSense", spawn_ball)
            key = cv2.waitKey(1)
            
            # quit if "q" or "esc" are pressed
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # stop the stream
        pipe.stop()
        
if __name__ == "__main__":
    main()