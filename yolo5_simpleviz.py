import time
import cv2
import numpy as np
import torch
from PIL import Image

from ouster.sdk import client
from ouster.sdk.client import ChanField, LidarScan
from ouster.sdk.client._utils import AutoExposure
from ouster.sdk import open_source

from ouster.sdk.viz import SimpleViz

SCAN_FPS = 10


def model_infer(model, frame):
    image = Image.fromarray(frame * 255)
    results = model(image)
    class_indices = results.pred[0][:, -1].int()
    return results.pred[0][class_indices == 0, :4]


def overlay_results(image, objects):
    for box in objects:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 1, 0), 1)


# create a post processor for every channel to perform auto leveling
post_processors = {
    ChanField.RANGE: AutoExposure(),
    ChanField.SIGNAL: AutoExposure(),
    ChanField.REFLECTIVITY: AutoExposure(),
    ChanField.NEAR_IR: AutoExposure()
}


def get_frame_from_scan(scan, channel):
    image = scan.field(channel).astype(np.float32)
    image = client.destagger(scan_source.metadata, image)
    post_processors[channel](image)
    return image


def process_scan(scan: LidarScan) -> LidarScan:
    global post_processors

    fields = [ChanField.SIGNAL, ChanField.REFLECTIVITY, ChanField.NEAR_IR]
    start = time.time()
    for f in fields:
        frame = get_frame_from_scan(scan, f)
        results = model_infer(model, frame)
        overlay_results(frame, results)
        # Stagger the frame again before storing the results again since
        # simpleviz will destagger the frames
        scan.field(f)[:] = client.destagger(scan_source.metadata,
                                            frame * 255, inverse=True)
    end = time.time()
    sleep_period = 1.0 / SCAN_FPS - (end - start)
    if sleep_period > 0:
        time.sleep(sleep_period)
    return scan


# Open the pcap file
pcap_file_path = "./Ouster-YOLOv5-sample.pcap"
scan_source = open_source(pcap_file_path, sensor_idx=0, cycle=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

scan_iterator = map(process_scan, scan_source)
SimpleViz(scan_source.metadata).run(scan_iterator)
scan_source.close()
