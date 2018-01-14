import cv2
import face_recognition
import sys
# import imutils
import glob
import os
import os.path

video_capture = cv2.VideoCapture(0)

BASE_FACES_FILE = 'base_faces'
face_image_files = glob.glob(os.path.join(BASE_FACES_FILE,"*"))
face_correct_names = []
face_encodings = []
face_locations = []
frame_face_encodings =[]

process_this_frame = True

for (i, face_image_file) in enumerate(face_image_files):
    filename = os.path.basename(face_image_file)
    face_correct_name = os.path.splitext(filename)[0]
    face_image = face_recognition.load_image_file(face_image_file)
    face_encoding =  face_recognition.face_encodings(face_image)[0]
    face_correct_names.append(face_correct_name)
    face_encodings.append(face_encoding)
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    for (face_correct_name, face_encoding) in zip(face_correct_names, face_encodings):
        # print(face_correct_name,face_encoding)
        if process_this_frame:
            frame_face_locations = face_recognition.face_locations(rgb_small_frame)
            frame_face_encodings = face_recognition.face_encodings(rgb_small_frame, frame_face_encodings)

            face_names = []
            for frame_face_encoding in face_encodings:
                print(face_encoding)
                match = face_recognition.compare_faces([face_encoding], frame_face_encoding)
# print(face_correct_names, face_encodings)
# weigao_image = face_recognition.load_image_file('./base_faces/weigao.JPG')
# weigao_face_encoding = face_recognition.face_encodings(weigao_image)[0]

# print(face_correct_names, face_encodings)
