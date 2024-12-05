from deepface import DeepFace
import realtime_face_recognition
import cv2
import sys
import os
import glob

#-------------------Variable Setting-------------------
# Input data source : "camera", "video", "image"
DATA_SOURCE = "camera"

# Camera(Webcam)
CAM_NUM = 0

# Image directory path
IMAGE_DIRECTORY_PATH = "data/face_samples/"
#DATABASE_DRECTORY_PATH = "data/face_database/"
DATABASE_DRECTORY_PATH = "customer_database/"
VIDEO_DIRECTORY_PATH = "data/video/"
# -----------------------------------------------------


class ImagePublisher():
    def __init__(self, data_source=DATA_SOURCE, cam_num=CAM_NUM, 
                 img_dir=IMAGE_DIRECTORY_PATH, db_path=DATABASE_DRECTORY_PATH, video_path=VIDEO_DIRECTORY_PATH,
                 recognizer=None):
        self.data_source = data_source
        self.cam_num = cam_num
        self.img_path = img_dir
        self.db_path = db_path
        self.video_path = video_path
        self.recognizer = recognizer

    def publish_images(self):
        if self.data_source == "camera":
            self.process_camera()
        elif self.data_source == "video":
            self.process_video()
        elif self.data_source == "image":
            self.process_image()
        else:
            print(f"Invalid data source: {self.data_source}")
            sys.exit(1)

    def setup_capture(self):
        self.cap = cv2.VideoCapture(self.cam_num)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print(f"Cannot open source: {self.data_source}")
            sys.exit(1)

    def process_camera(self):
        #self.setup_capture()
        self.recognizer.perform_streaming()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Camera frame", frame)
                       
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()

    def process_video(self):
        self.setup_capture()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("Video frame", frame)
            self.recognizer.perform_face_recognition(frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.cap.release()
    
    def process_image(self):
        if os.path.isdir(self.img_path):
            self.img_list = glob.glob(self.img_path + "*")
            self.img_num = 0
            print(">> Image List:", self.img_list)  
        else:
            print(f"Not a directory file: {self.img_path}")
            sys.exit(1)

        while self.img_num < len(self.img_list):
            img_file = self.img_list[self.img_num]
            img = cv2.imread(img_file)  
            if img is None:
                print(f"Skipping non-image file : {img_file}")
            else:
                print(f"Pulished image: {img_file}")
                cv2.imshow("Pulished image", img)
                self.recognizer.perform_face_recognition(img)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
            self.img_num += 1


class FaceRecognition():
    def __init__(self, db_path=DATABASE_DRECTORY_PATH):
        self.models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        self.backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.db_path = db_path

    def perform_face_recognition(self, img):
        try:
            people = DeepFace.find(img_path=img,
                                   db_path=self.db_path,
                                   model_name=self.models[2],
                                   detector_backend=self.backends[4],
                                   distance_metric=self.metrics[2],
                                   enforce_detection=False)
            people = people[0]
            for idx, person in people.iterrows():
                img_db = cv2.imread(person["identity"])
                result = self.highlight_facial_areas(img_db, person)

        except Exception as e:
            print(f"Error in face recognition: {e}")
            sys.exit(1)

        cv2.imshow("Face recognition", result)

    def highlight_facial_areas(self, img, person):
        name = person["identity"].split("/")[3]
        threshold = person["threshold"]
        distance = person["distance"]
        print("Database >> name: %s, threshold: %s, distance: %s" % (name, threshold, distance))

        # Bounding box coordinates for the facial region 
        x = person["target_x"]
        y = person["target_y"]
        w = person["target_w"]
        h = person["target_h"]
        

        # Draw a bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(img, f"Name: {name}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Distance: {distance}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img
    
    def perform_streaming(self):
        """
        result = DeepFace.stream(db_path=self.db_path,
                                 model_name=self.models[2],
                                 detector_backend=self.backends[3],
                                 distance_metric=self.metrics[1],
                                 time_threshold=3
                                 )
        """
        result = realtime_face_recognition.analysis(db_path=self.db_path,
                                 model_name=self.models[2],
                                 detector_backend=self.backends[3],
                                 distance_metric=self.metrics[1],
                                 time_threshold=3
                                 )
        


def main():
    print("Check image publish")
    recognizer = FaceRecognition()
    pub = ImagePublisher(recognizer=recognizer)
    pub.publish_images()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()