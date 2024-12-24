import os
from deepface.commons.logger import Logger

logger = Logger()

def initialize_dir():
    home = str(os.getenv("F2M_HOME", default=os.path.expanduser("~")))
    f2m_home_path = os.path.join(home, "Fruit-Flow-Market")
    face_detector_path = os.path.join(f2m_home_path, "AIServer/FaceDetector")
    billing_gui_path = os.path.join(f2m_home_path, "FaceDevice/BillingGUI")
    member_db_path = os.path.join(face_detector_path, "member_database")

    if not os.path.exists(member_db_path):
        os.makedirs(member_db_path, exist_ok=True)
        logger.info(f"Directory {member_db_path} has been created")
    
    return member_db_path