# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 15:00:47 2021

@author: surya
"""

import cv2
import numpy as np
import time

class Obj_detector():
    def __init__(self, save_vid_path='output.mp4', fps=30, size=(352,640), weights='yolov3_last.weights', cfg='yolov3_custom.cfg', name_file='obj.names'):       
        self.cap = cap
        self.save_vid_path = save_vid_path
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.fps = fps
        self.size = size
        self.weights = weights
        self.cfg = cfg
        self.name_file = name_file       
        with open(self.name_file, 'r') as f:
            self.classes = f.read().splitlines()    



    def Detect(self, img, crop=False, con_thresh=0.5, nms_thresh=0.5):
        
        net = cv2.dnn.readNet(self.weights, self.cfg)
        
        blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=crop)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
    
        #img = cv2.resize(img, (352,640))
        h, w, _ = img.shape
        boxes = []
        confidences = []
        class_ids = []
    
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                if confidence > con_thresh:
                    center_x = int(detection[0]*w)
                    center_y = int(detection[1]*h)
                    b_w = int(detection[2]*w)
                    b_h = int(detection[3]*h)
                    x = int(center_x - b_w/2)
                    y = int(center_y - b_h/2)
                    
                    boxes.append([x,y,b_w,b_h])  #Bounding boxes details  
                    confidences.append(confidence)   #Condidence level for each corresponding bboxes
                    class_ids.append(class_id)   #Class assigned to each corresponding bbox by their ids   here, (Human Hands - 1, Human Face - 2)
    
        print(len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, con_thresh, nms_thresh)   #Gives indices of those boxes which are left out after eliminating redundent boxes
        
        return indexes, boxes, confidences, class_ids
        
    
    
    def Draw_Bbox(self, img, indexes, boxes, confidences, class_ids, draw=True):
        
        if draw:
            try:
                print(indexes.flatten())
                colors = np.random.uniform(0, 255, size=(len(boxes), 3))
                
                for i in indexes.flatten():
                    x, y, b_w, b_h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    c = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+b_w, y+b_h), color, 2)
                    cv2.putText(img, label + ' ' + c, (x, y+20), self.font, 2, (0,255,0), 2)
                    print(x,y,b_w,b_h)
            
            except:
                pass

def main(capture = 0, weights='yolov3_last.weights', cfg='yolov3_custom.cfg'):
    ptime = 0
    ctime = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    detector = Obj_detector(weights=weights, cfg=cfg)
    out_vid = cv2.VideoWriter(detector.save_vid_path, fourcc, detector.fps, detector.size)
    
    cap = cv2.VideoCapture(capture)
    
    while True:
        
        _,img = cap.read()
        indexes, boxes, confidences, class_ids = detector.Detect(img)
        detector.Draw_Bbox(img, indexes, boxes, confidences, class_ids)
        
        ctime = time.time()
        current_fps = 1/(ctime-ptime)
        ptime = ctime
        
        out_vid.write(img)
        cv2.putText(img, str(int(current_fps)), (int(0.1*detector.size[0]), int(0.1*detector.size[1])), detector.font, 3, (255,0,255), 3)
        cv2.imshow('vid', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    cap = input('Enter webcam(0) or mp4 video(file_name) : ')
    if cap=='0':
        cap = int(cap)
    k = input('Do you want to add custom cfg and weight files?(y/n) : ')
    if k=='y':
        w = input('Enter weight file: ')
        c = input('Enter cfg file: ')
        main(cap,w,c)
    else:
        main(cap)