import time
import math
import cv2
import numpy as np
import torch
from PIL import Image

from ouster.sdk import client
from ouster.sdk.client import ChanField
from ouster.sdk.client._utils import AutoExposure
from ouster.sdk import open_source


def model_infer(model, frame):
    image = Image.fromarray(frame * 255)
    results = model(image)
    class_indices = results.pred[0][:, -1].int()
    return results.pred[0][class_indices == 0, :4]


def draw_results(image, objects, sensor_info, scan, xyzlut):
    """draw detected persons with distance"""
    xyz_destaggered = client.destagger(sensor_info, xyzlut(scan))
    range_val = scan.field(ChanField.RANGE).astype(np.float32)

    for box in objects:
        x1, y1, x2, y2 = map(int, box)
        range_roi = range_val[y1:y2, x1:x2]
        # only consider points /w valid returns
        valid_returns_idx = np.where(range_roi > 0)
        poi_roi = np.unravel_index(range_roi[valid_returns_idx].argmin(),
                                   range_roi.shape)  # (y,x) in roi
        poi_x = poi_roi[1] + x1
        poi_y = poi_roi[0] + y1
        poi = (poi_y, poi_x)  # (y,x) in global
        xyz = xyz_destaggered[poi]
        dist = math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 1, 0), 1)
        cv2.putText(image, F"person {round(dist,2)} m", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 1, 0), 1, 2)


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


def get_scan_size_and_fps(sensor_info: client.SensorInfo):
    """extract the scan width x heigh x fps information from the json file"""
    w = sensor_info.mode.cols
    h = len(sensor_info.beam_altitude_angles)
    fps = sensor_info.mode.frequency
    return w, h, fps


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the pcap file
pcap_file_path = "./Ouster-YOLOv5-sample.pcap"

scan_source = open_source(pcap_file_path, sensor_idx=0, cycle=True)
scan_width, scan_height, scan_fps = get_scan_size_and_fps(scan_source.metadata)

video_fps = scan_fps    # apply the same fps to the video size
video_size = (scan_width, scan_height * 4)  # We multiply the video output by 4
# since we are going to stack the 4
# frames on top of each others vertically

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, video_fps, video_size)

xyzlut = client.XYZLut(scan_source.metadata)  # construct the cartesian lookup table

def run():

    fields = [ChanField.RANGE, ChanField.SIGNAL,
              ChanField.REFLECTIVITY, ChanField.NEAR_IR]
    images = [None] * len(fields)

    for scan in scan_source:
        # process the scan
        start = time.time()
        for i in range(len(fields)):
            images[i] = get_frame_from_scan(scan, fields[i])
            results = model_infer(model, images[i])
            # convert to a colored image
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
            draw_results(images[i], results,
                         scan_source.metadata, scan, xyzlut)
        end = time.time()
        sleep_period = 1.0 / scan_fps - (end - start)
        if sleep_period > 0:
            time.sleep(sleep_period)

        stacked_images = cv2.vconcat(images)
        converted_image = cv2.convertScaleAbs(stacked_images * 255)
        cv2.imshow('Lidar Detection', stacked_images)
        out.write(converted_image)

        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
            # Press 'ESC' to exit the loop
            if key == 27:
                return


run()
scan_source.close()
out.release()
cv2.destroyAllWindows()
