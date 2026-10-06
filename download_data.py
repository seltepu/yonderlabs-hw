import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set ROBOFLOW_API_KEY in your environment or .env file")

rf = Roboflow(api_key=api_key)
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
dataset = project.version(2).download("yolov8")
print("Downloaded to:", dataset.location)