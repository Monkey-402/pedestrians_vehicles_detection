from roboflow import Roboflow
rf = Roboflow(api_key="Os6qrkPAXImdv8jaBJNR")
project = rf.workspace("person-and-vehicles-road-cctv").project("person-and-vehicle")
version = project.version(13)
dataset = version.download("yolov11")
                