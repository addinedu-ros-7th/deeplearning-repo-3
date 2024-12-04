# built-in dependencies
import os
import time
from typing import List, Tuple, Optional
import traceback

# 3rd party dependencies
import numpy as np
import pandas as pd
import cv2

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


IDENTIFIED_IMG_SIZE = 112
TEXT_COLOR = (255, 255, 255)

# pylint: disable=unused-variable
def analysis(
    db_path: str,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=3,
    frame_threshold=5,
    anti_spoofing: bool = False,
):
    # initialize models
    build_facial_recognition_model(model_name=model_name)
    # call a dummy find function for db_path once to create embeddings before starting webcam
    _ = search_identity(
        detected_face=np.zeros([224, 224, 3]),
        db_path=db_path,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        model_name=model_name,
    )

    freezed_img = None
    freeze = False
    num_frames_with_faces = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    while True:
        has_frame, img = cap.read()
        logger.info("Show your face")
        if not has_frame:
            break

        raw_img = img.copy()

        faces_coordinates = []
        if freeze is False:
            faces_coordinates = grab_facial_areas(
                img=img, detector_backend=detector_backend, anti_spoofing=anti_spoofing
            )

            detected_faces = extract_facial_areas(img=img, faces_coordinates=faces_coordinates)

            img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
            img = countdown_to_freeze(
                img=img,
                faces_coordinates=faces_coordinates,
                frame_threshold=frame_threshold,
                num_frames_with_faces=num_frames_with_faces,
            )

            num_frames_with_faces = num_frames_with_faces + 1 if len(faces_coordinates) else 0

            freeze = num_frames_with_faces > 0 and num_frames_with_faces % frame_threshold == 0
            if freeze:
                # add analyze results into img - derive from raw_img
                img = highlight_facial_areas(
                    img=raw_img, faces_coordinates=faces_coordinates, anti_spoofing=anti_spoofing
                )
                
                # facial recogntion analysis
                img, target_name = perform_facial_recognition(
                    img=img,
                    faces_coordinates=faces_coordinates,
                    detected_faces=detected_faces,
                    db_path=db_path,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    model_name=model_name,
                )

                print("User Info------------")
                print("name : ", target_name)
                print("---------------------")

                # freeze the img after analysis
                freezed_img = img.copy()

                # start counter for freezing
                tic = time.time()
                logger.info("freezed")

        elif freeze is True and time.time() - tic > time_threshold:
            freeze = False
            freezed_img = None
            # reset counter for freezing
            tic = time.time()
            logger.info("freeze released")

        freezed_img = countdown_to_release(img=freezed_img, tic=tic, time_threshold=time_threshold)

        cv2.imshow("img", img if freezed_img is None else freezed_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()

    return target_name


def build_facial_recognition_model(model_name: str) -> None:
    _ = DeepFace.build_model(task="facial_recognition", model_name=model_name)
    logger.info(f"{model_name} is built")


#Search an identity in facial database.
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
            silent=True,
        )

        # print(dfs)
    except ValueError as err:
        if f"No item found in {db_path}" in str(err):
            logger.warn(
                f"No item is found in {db_path}."
                "So, no facial recognition analysis will be performed."
            )
            dfs = []
        else:
            raise err
    if len(dfs) == 0:
        # you may consider to return unknown person's image here
        return None, None

    # detected face is coming from parent, safe to access 1st index
    df = dfs[0]

    print("Recognized Faces-----")
    print(df)
    print("---------------------")

    if df.shape[0] == 0:
        return None, None

    candidate = df.iloc[0]
    target_path = candidate["identity"]
    #print(type(target_path))
    target_name = target_path.split("/")[1]
    logger.info(f"Hello, {target_name}")

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

    return target_path.split("/")[-1], target_img, target_name

"""
def build_demography_models(enable_face_analysis: bool) -> None:
    if enable_face_analysis is False:
        return
    DeepFace.build_model(task="facial_attribute", model_name="Age")
    logger.info("Age model is just built")
    DeepFace.build_model(task="facial_attribute", model_name="Gender")
    logger.info("Gender model is just built")
    DeepFace.build_model(task="facial_attribute", model_name="Emotion")
    logger.info("Emotion model is just built")
"""

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
    # do not take any action if it is not frozen yet
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
            # you may consider to extract with larger expanding value
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
    except:  # to avoid exception if no face detected
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
        target_label, target_img, target_name= search_identity(
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

        # print("User Info : ", target_name)

    return img, target_name




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

