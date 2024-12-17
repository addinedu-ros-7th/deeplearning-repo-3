# built-in dependencies
import os
import time
from typing import List, Tuple, Optional
import traceback

# 3rd party dependencies
import numpy as np
import pandas as pd
import cv2
import re

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# project dependencies
from deepface import DeepFace
from commons.logger import logger
from commons import dir_utils
from modules.modeling import modeling

logger = logger()

IDENTIFIED_IMG_SIZE = 112
TEXT_COLOR = (255, 255, 255)


class RecognitionHandler():
    def __init__(self, db_path, model_name, detector_backend, distance_metric, time_threshold):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.time_threshold = time_threshold
        
        self.camera = None
        self.initialized = False
        self.target_id = None
        self.target_name = None
        self.running = True

        # status various
        self.freeze = False
        self.tic = time.time()
        self.num_frames_with_faces = 0
        self.freezed_img = None        

        self.initialize()

    def initialize(self):
        if not self.initialized:
            build_facial_recognition_model(model_name=self.model_name)
            logger.info("Model initialized successfully")

            # call a dummy find function for db_path once to create embeddings before starting webcam
            _ = search_identity(
                detected_face=np.zeros([224, 224, 3]),
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
            )
            logger.info(f"Database embeddings created successfully: {self.db_path}")
            self.initialized = True
        else:
            logger.info("Recognition module is already initialized")

        if self.camera is None:
            logger.info("Camera is starting")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

    def analysis(
        self,
        db_path: str,
        model_name="VGG-Face",
        detector_backend="opencv",
        distance_metric="cosine",
        enable_face_analysis=True,
        source=0,
        time_threshold=5,
        frame_threshold=5,
        anti_spoofing: bool = False,
    ):
        logger.info("Analysis is starting")
        while self.running:
            has_frame, img = self.cap.read()
            if not has_frame:
                logger.warning("Failed to capture frame from the camera. Check camera connection or configuration.")
                break
            
            raw_img = img.copy()

            faces_coordinates = []
            if self.freeze is False:
                faces_coordinates = grab_facial_areas(
                    img=img, detector_backend=detector_backend, anti_spoofing=anti_spoofing
                )

                detected_faces = extract_facial_areas(img=img, faces_coordinates=faces_coordinates)
                img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)

                img = countdown_to_freeze(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    frame_threshold=frame_threshold,
                    num_frames_with_faces=self.num_frames_with_faces,
                )

                self.num_frames_with_faces = self.num_frames_with_faces + 1 if len(faces_coordinates) else 0

                self.freeze = self.num_frames_with_faces > 0 and self.num_frames_with_faces % frame_threshold == 0
                if self.freeze:
                    logger.info("Face detected. Starting facial recognition analysis")
                    # add analyze results into img - derive from raw_img
                    img = highlight_facial_areas(
                        img=raw_img, faces_coordinates=faces_coordinates, anti_spoofing=anti_spoofing
                    )

                    # facial recogntion analysis
                    img, self.target_id, self.target_name = perform_facial_recognition(
                        img=img,
                        faces_coordinates=faces_coordinates,
                        detected_faces=detected_faces,
                        db_path=self.db_path,
                        detector_backend=self.detector_backend,
                        distance_metric=self.distance_metric,
                        model_name=self.model_name,
                    )

                    #if self.target_id:
                    #    logger.info(f"member : {self.target_id} {self.target_name}")

                    # freeze the img after analysis
                    self.freezed_img = img.copy()

                    # start counter for freezing
                    self.tic = time.time()
                    logger.info("Image frozen for feedback (Freeze duration: %d seconds)", time_threshold)

            elif self.freeze is True and time.time() - self.tic > time_threshold:
                self.freeze = False
                self.freezed_img = None
                # reset counter for freezing
                self.tic = time.time()
                logger.info("Resuming face detection")

            self.freezed_img = countdown_to_release(img=self.freezed_img, tic=self.tic, time_threshold=time_threshold)

            #cv2.imshow("img", img if self.freezed_img is None else self.freezed_img)
            #if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            #    self.running = False
            #    logger.info("releasing camera resources and shutting down")
            #    break
            
            return self.target_id, self.target_name, img if self.freezed_img is None else self.freezed_img
            
        # kill open cv things
        self.cap.release()
        #return img if freezed_img is None else freezed_img


def build_facial_recognition_model(model_name: str) -> None:
    _ = DeepFace.build_model(task="facial_recognition", model_name=model_name)
    logger.info(f"{model_name} is built")


# Search an identity in facial database.
def search_identity(
    detected_face: np.ndarray,
    db_path: str,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    target_path = None
    try:
        dfs = DeepFace.find(
            img_path=detected_face,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
            threshold=18,
            silent=True
        )

        #print(dfs)
    except ValueError as err:
        if f"No item found in {db_path}" in str(err):
            logger.warning(
                f"No item is found in {db_path}."
                "So, no facial recognition analysis will be performed."
            )
            dfs = []
        else:
            raise err

    df = dfs[0]

    if len(df) == 0:
        logger.info("Search identify: Unknown")
        return None, detected_face, "Unknown", "Unknown"
    #print("Recognized Faces-----")
    #print(df)
    #print("---------------------")

    if df.shape[0] == 0:
        return None, None, None, None

    candidate = df.iloc[0]
    target_path = candidate["identity"]
    #print(type(target_path))
    target_id = target_path.split("/")[7]
    filename = target_path.split("/")[8]
    target_name = re.sub(r"\d+\.jpg$", "", filename)
    threshold = candidate["threshold"]
    distance = candidate["distance"]
    logger.info(f"Search identify : {target_id} {target_name} ( {threshold} / {distance} )")

    # load found identity image - extracted if possible
    target_objs = DeepFace.extract_faces(
        img_path=target_path,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=True,
    )

    # extract facial area of the identified image if and only if it has one face
    # otherwise, show image as is
    if len(target_objs) == 1:
        # extract 1st item directly
        target_obj = target_objs[0]
        target_img = target_obj["face"]
        target_img *= 255
        target_img = target_img[:, :, ::-1]
    else:
        target_img = cv2.imread(target_path)

    # resize anyway
    target_img = cv2.resize(target_img, (IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE))

    return target_path.split("/")[-1], target_img, target_id, target_name


def highlight_facial_areas(
    img: np.ndarray,
    faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
    anti_spoofing: bool = False,
) -> np.ndarray:
    for x, y, w, h, is_real, antispoof_score in faces_coordinates:
        # highlight facial area with rectangle

        if anti_spoofing is False:
            color = (67, 67, 67)
        else:
            if is_real is True:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    return img


def countdown_to_freeze(
    img: np.ndarray,
    faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
    frame_threshold: int,
    num_frames_with_faces: int,
) -> np.ndarray:
    for x, y, w, h, is_real, antispoof_score in faces_coordinates:
        cv2.putText(
            img,
            str(frame_threshold - (num_frames_with_faces % frame_threshold)),
            (int(x + w / 4), int(y + h / 1.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (255, 255, 255),
            2,
        )
    return img


def countdown_to_release(
    img: Optional[np.ndarray], tic: float, time_threshold: int
) -> Optional[np.ndarray]:
    if img is None:
        return img
    toc = time.time()
    time_left = int(time_threshold - (toc - tic) + 1)
    cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -10)
    cv2.putText(
        img,
        str(time_left),
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
    )
    return img


def grab_facial_areas(
    img: np.ndarray, detector_backend: str, threshold: int = 130, anti_spoofing: bool = False
) -> List[Tuple[int, int, int, int, bool, float]]:
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=detector_backend,
            expand_percentage=0,
            anti_spoofing=anti_spoofing,
        )
        faces = [
            (
                face_obj["facial_area"]["x"],
                face_obj["facial_area"]["y"],
                face_obj["facial_area"]["w"],
                face_obj["facial_area"]["h"],
                face_obj.get("is_real", True),
                face_obj.get("antispoof_score", 0),
            )
            for face_obj in face_objs
            if face_obj["facial_area"]["w"] > threshold
        ]
        return faces
    except:
        return []


def extract_facial_areas(
    img: np.ndarray, faces_coordinates: List[Tuple[int, int, int, int, bool, float]]
) -> List[np.ndarray]:
    detected_faces = []
    for x, y, w, h, is_real, antispoof_score in faces_coordinates:
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
        detected_faces.append(detected_face)
    return detected_faces


def perform_facial_recognition(
    img: np.ndarray,
    detected_faces: List[np.ndarray],
    faces_coordinates: List[Tuple[int, int, int, int, bool, float]],
    db_path: str,
    detector_backend: str,
    distance_metric: str,
    model_name: str,
) -> np.ndarray:
    for idx, (x, y, w, h, is_real, antispoof_score) in enumerate(faces_coordinates):
        detected_face = detected_faces[idx]
        target_label, target_img, target_id, target_name=search_identity(
            detected_face=detected_face,
            db_path=db_path,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            model_name=model_name,
        )
        if target_label is None:
            continue

        img = overlay_identified_face(
            img=img,
            target_img=target_img,
            label=target_label,
            x=x,
            y=y,
            w=w,
            h=h,
        )
    return img, target_id, target_name


def overlay_identified_face(
    img: np.ndarray,
    target_img: np.ndarray,
    label: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> np.ndarray:
    try:
        if y - IDENTIFIED_IMG_SIZE > 0 and x + w + IDENTIFIED_IMG_SIZE < img.shape[1]:
            # top right
            img[
                y - IDENTIFIED_IMG_SIZE : y,
                x + w : x + w + IDENTIFIED_IMG_SIZE,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x + w, y),
                (x + w + IDENTIFIED_IMG_SIZE, y + 20),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x + w, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y),
                (x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (x + 3 * int(w / 4), y - int(IDENTIFIED_IMG_SIZE / 2)),
                (x + w, y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif y + h + IDENTIFIED_IMG_SIZE < img.shape[0] and x - IDENTIFIED_IMG_SIZE > 0:
            # bottom left
            img[
                y + h : y + h + IDENTIFIED_IMG_SIZE,
                x - IDENTIFIED_IMG_SIZE : x,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x - IDENTIFIED_IMG_SIZE, y + h - 20),
                (x, y + h),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x - IDENTIFIED_IMG_SIZE, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) - int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif y - IDENTIFIED_IMG_SIZE > 0 and x - IDENTIFIED_IMG_SIZE > 0:
            # top left
            img[y - IDENTIFIED_IMG_SIZE : y, x - IDENTIFIED_IMG_SIZE : x] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x - IDENTIFIED_IMG_SIZE, y),
                (x, y + 20),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x - IDENTIFIED_IMG_SIZE, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y),
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) - int(w / 4),
                    y - int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x, y - int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )

        elif (
            x + w + IDENTIFIED_IMG_SIZE < img.shape[1]
            and y + h + IDENTIFIED_IMG_SIZE < img.shape[0]
        ):
            # bottom righ
            img[
                y + h : y + h + IDENTIFIED_IMG_SIZE,
                x + w : x + w + IDENTIFIED_IMG_SIZE,
            ] = target_img

            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(
                img,
                (x + w, y + h - 20),
                (x + w + IDENTIFIED_IMG_SIZE, y + h),
                (46, 200, 255),
                cv2.FILLED,
            )
            cv2.addWeighted(
                overlay,
                opacity,
                img,
                1 - opacity,
                0,
                img,
            )

            cv2.putText(
                img,
                label,
                (x + w, y + h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
            )

            # connect face and text
            cv2.line(
                img,
                (x + int(w / 2), y + h),
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (67, 67, 67),
                1,
            )
            cv2.line(
                img,
                (
                    x + int(w / 2) + int(w / 4),
                    y + h + int(IDENTIFIED_IMG_SIZE / 2),
                ),
                (x + w, y + h + int(IDENTIFIED_IMG_SIZE / 2)),
                (67, 67, 67),
                1,
            )
        else:
            logger.info("cannot put facial recognition info on the image")
    except Exception as err:  # pylint: disable=broad-except
        logger.error(f"{str(err)} - {traceback.format_exc()}")
    return img

