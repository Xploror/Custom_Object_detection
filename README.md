# Custom_Object_detection
This repository contains object detection module based on custom trained darknet models trained under custom dataset and then using the module, object detection can be implemented on any image, video or webcam.

---------

## Custom Data Folder

It contains data, names and cfg files which were used to custom train a Yolo model. Here there are 2 custom cfg files corresponding to YOLOv3 and YOLOv3_tiny as both were used to test the accuracy and speed of object detection. YOLOv3 is slow but very accurate compared to YOLOv3_tiny where, speed is much faster but not so accurate. **Store your cfg files in this folder and whenever you need to access it from the module then draw that perticular cfg file just out of the folder or else change the default address of the cfg file in the module pointing to the file inside the folder**.

## customTrackYOLO

This is the main module for implementing custom object detection on any image, video or webcam. It uses dnn method inside the cv2 package to read the weight files with the given cfg file and forward the weights into the cfg directed model with input as blob of the image. This is done using the class Obj_detector() under which there are methods like Detect() and Draw_Bbox() to detect in a given frame of image and whether to make bounding boxes around the object or not, respectively.

**Also required is the weights file which you got after custom training the YOLO model. It is required by the dnn method for reference. Check out how to do custom YOLO training with GPU enabled in Google Collaboratory here :**

https://github.com/Xploror/Object_Detection_YOLO

## Test examples

https://github.com/user-attachments/assets/ec968e76-84da-4727-aa2a-ea121c2305ac




https://github.com/user-attachments/assets/b7a62619-d4d0-41f7-8645-2a0ee43d2d41


