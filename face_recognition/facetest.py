import face_recognition as fr
import numpy as np

class FaceGroup:
    def __init__(self, face_images):
        self.face_images=face_images
        self.face_encodings = [fr.face_encodings(fr.load_image_file(img.fname))[0] for img in face_images]


    def detect(self, image):
        encoding = fr.face_encodings(fr.load_image_file(image))
        if(len(encoding)<1):
            return None
        result =np.where(fr.compare_faces(self.face_encodings, fr.face_encodings(fr.load_image_file(image))[0], 0.4))[0]
        return [self.face_images[i].id for i in result]


class FaceImage:
    def __init__(self, id, fname):
        self.id=id
        self.fname=fname

face_data = [FaceImage(face_id, 'C:\\Users\\Administrator\\Pictures\\face_recognition\\data\\%d.jpg'%(face_id, )) for face_id in range(1,11) ]
test_files = ['C:\\Users\\Administrator\\Pictures\\face_recognition\\test\\%d.jpg'%(face_id, ) for face_id in range(1,11)]

group=FaceGroup(face_data)
for t in test_files:
    print(group.detect(t))

