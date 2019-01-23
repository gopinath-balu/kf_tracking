#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: ambakick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import detector
from mtcnn.mtcnn import MTCNN
import tracker
import cv2
import copy

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 3  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits = 1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque([str(i) for i in range(1, 1000000)])

tracker_data = {}

base_path = os.getcwd()
out_folder = os.path.join(base_path, 'people_folder')

debug = False

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers), len(detections)), dtype=np.float32)
    # print('IOU_mat {} , trackers {} , detections {}'.format(IOU_mat, len(trackers), len(detections)))
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = helpers.box_iou2(trk,det[0]) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)
    # print('original {}, shape of orig {} '.format(linear_assignment(-IOU_mat), linear_assignment(-IOU_mat).shape))        

    unmatched_trackers, unmatched_detections = [], []
    
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global matchesin_hits
    global track_id_list
    global debug


    # Check if output folder exists if not create
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    frame_count+=1
    
    
    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img) # measurement
    # print(z_box)
    if debug:
       print('Frame:', frame_count)

    x_box =[]

    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(img, z_box[i][0], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    
    # print('len(tracker_list), tracker_list ', len(tracker_list), tracker_list)

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    # print('x_box, z_box ', x_box, z_box)
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
    #print('matched, unmatched_dets, unmatched_trks', matched, unmatched_dets, unmatched_trks)
    # print('matched, ', matched)

    # Deal with matched detections     
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx][0]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker_list[trk_idx]
            # print(type(tmp_trk))
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.det_conf = z_box[det_idx][1]
    
    # Deal with unmatched detections      
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_box[idx][0]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            # print(tmp_trk.id)

            tracker_list.append(tmp_trk)
            x_box.append(xx)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks) > 0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx



    # The list of tracks to be annotated  
    good_tracker_list = []
    
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.box
            conf = trk.det_conf
            if debug:
                print('updated box: ', x_cv2)
                print()
            img_to_write = copy.deepcopy(img)
            img = helpers.draw_box_label(trk.id, img, x_cv2) # Draw the bounding boxes on the images
            
            # write_img = img[x_cv2[0]:x_cv2[2], x_cv2[1]:x_cv2[3]]
            # cv2.imwrite(os.path.join(out_folder, str(trk.id) + '_' + str(conf) + '.jpg'), write_img)


    for trk in good_tracker_list:
        
        # print('trk.id, trk.box, trk.det_conf  ------ > ', trk.id, trk.box, trk.det_conf)
        
        if (trk.id not in tracker_data.keys()):
            tracker_data[trk.id] = [conf, x_cv2, img_to_write]
            print('not in tracker_data add new data')

        if (trk.id in tracker_data.keys()):
            print('data in tracker_data but conf low so no change')
            if (trk.det_conf > tracker_data[trk.id][0]):
                tracker_data[trk.id] = [conf, x_cv2, img_to_write]
                print(' changed tracker_data')


    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)  
    
    # Write tracker images to file
    for d_trk in deleted_tracks:
        if(d_trk.id in tracker_data.keys()):
            print('delected track matched here so track is ', d_trk.id, tracker_data[d_trk.id][0])
            y1, y2 = tracker_data[d_trk.id][1][0], tracker_data[d_trk.id][1][2]
            x1, x2 = tracker_data[d_trk.id][1][1], tracker_data[d_trk.id][1][3]
            print('y1, y2, x1, x2 ', y1, y2, x1, x2)
            trk_id = d_trk.id
            print('trk_id ', trk_id)
            trk_conf = tracker_data[d_trk.id][0]
            print('trk_conf ', trk_conf)
            trk_img = tracker_data[d_trk.id][2]
            print('trk_img ', type(trk_img), trk_img.shape)
            write_img = trk_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(out_folder, str(trk_id) + '_' + str(trk_conf) + '.jpg'), write_img)
            del tracker_data[d_trk.id]

    

    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses <= max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
    cv2.imshow("frame", img)
    return img
    
if __name__ == "__main__":    
    
    det = detector.PersonDetector()
    # det = MTCNN()

    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        
        for i in range(len(images))[0:7]:
             image = images[i]
             image_box = pipeline(image)   
             plt.imshow(image_box)
             plt.show()
           
    else: # test on a video file.
        
        # start=time.time()
        # # output = 'test_v7.mp4'
        # clip1 = VideoFileClip("project_video.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        # clip = clip1.fl_image(pipeline)
        # clip.write_videofile(output, audio=False)
        # end  = time.time()
        # cap = cv2.VideoCapture("http://root:axis0235@10.0.4.200/mjpg/1/video.mjpg")
        cap = cv2.VideoCapture(0)

        # cap = cv2.VideoCapture('test_file.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 8.0, (640,480))

        while(True):
            
            ret, img = cap.read()
            img = cv2.resize(img, (640, 480))
            # print(img, type(img))
            
            np.asarray(img)
            new_img = pipeline(img)
            out.write(new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        # print(round(end-start, 2), 'Seconds to finish')
