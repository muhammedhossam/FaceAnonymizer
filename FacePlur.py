import cv2
import mediapipe as mp
import os
import argparse
from Utils import *

# make argument and call mode  and the filePath
args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default='Video.mp4')
args = args.parse_args()

# get the out path
output_path = './output'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# make instance of the model
mp_face = mp.solutions.face_detection

# play maker
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    if args.mode in ['image']:

        # read the image
        img = cv2.imread(args.filePath)

        # play
        img = process_img(img,face_detection = face_detection)

        # save the image
        cv2.imwrite(os.path.join(output_path, 'face_blured.jpg'), img)
    # For the Video
    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)

        # get the video and read
        ret, frame = cap.read()

        # read and write the output frame by frame with that instance
        out_video = cv2.VideoWriter(os.path.join(output_path, 'out_video.mp4'),
                                    cv2.VideoWriter_fourcc(*'MP4V'),
                                    25,
                                    (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection=face_detection)
            out_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        out_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        while ret:
            frame = process_img(frame, face_detection=face_detection)
            cv2.imshow('Webcam', frame)
            cv2.waitKey(25)
            ret, frame = cap.read()
        cap.release()
