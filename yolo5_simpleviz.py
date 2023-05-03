import time
import cv2
import numpy as np
import torch
from PIL import Image
import itertools

from ouster import client
from ouster.client import ChanField, LidarScan
from ouster.pcap import Pcap
from ouster.sdk.util import resolve_metadata
from ouster.client._utils import AutoExposure

from ouster.sdk.simple_viz import SimpleViz

class ScanCapture:
    def __init__(self, source, loop=True) -> None:
        self._open_pcap(source)
        self._loop = loop
    @property
    def sensor_info(self):
        return self._info
    @property
    def scans_loop(self) -> LidarScan:
        return self._scans_loop
    def _open_pcap(self, file_path):
        meta_file = resolve_metadata(file_path)
        with open(meta_file, "r") as f:
            self._info = client.SensorInfo(f.read())
        if self._info:
            self._scans = client.Scans(Pcap(file_path, self._info), complete=True)
        else:
            print("error opening the stream")
        # Use this only on short clips
        self._scans_loop = itertools.cycle(self._scans)
    def _read(self):
        if self._loop == True:
            for s in self._scans_loop:
                yield s
        else:
            for s in self._scans:
                yield s
    def read(self) -> LidarScan:
        try:
            return True, next(self._read())
        except Exception as e:
            print(e)
            return False, None
    def release(self):
        self._scans.close()

# Open the pcap file
pcap_file_path = "./Ouster-YOLOv5-sample.pcap"
scan_capture = ScanCapture(pcap_file_path, loop=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def model_infer(model, frame):
    image = Image.fromarray(frame * 255)
    results = model(image)
    class_indices = results.pred[0][:, -1].int()
    return results.pred[0][class_indices == 0, :4]

def draw_results(image, objects):
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
    image = client.destagger(scan_capture.sensor_info, image)
    post_processors[channel](image)
    return image


# extract the scan width x heigh x fps information from the json file
def get_scan_size_and_fps(metadata_path):
    with open(metadata_path, 'r') as file:
        import json
        data = json.load(file)
        lidar_mode = data["lidar_mode"].split('x')
        w = int(lidar_mode[0])
        h = len(data["beam_altitude_angles"])
        fps = int(lidar_mode[1])
        return w, h, fps
    
scan_width, scan_height, scan_fps = get_scan_size_and_fps('./Ouster-YOLOv5-sample.json')


def process_scan(scan: LidarScan) -> LidarScan:
    global post_processors

    fields = [ChanField.SIGNAL, ChanField.REFLECTIVITY, ChanField.NEAR_IR]

    for  f in fields:
        frame = get_frame_from_scan(scan, f)
        results = model_infer(model, frame)
        draw_results(frame, results)
        # Stagger the frame again before storing the results again
        scan.field(f)[:] = client.destagger(scan_capture.sensor_info, frame * 255, inverse=True)

    time.sleep(0.033)

    return scan

scan_iterator = map(process_scan, scan_capture.scans_loop)
SimpleViz(scan_capture.sensor_info).run(scan_iterator)

scan_capture.release()
