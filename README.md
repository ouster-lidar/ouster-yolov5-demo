# ouster-yolov5-demo

- [Running the demo on Google colab](#running-the-demo-on-google-colab)
- [Running the demo locally](#running-the-demo-locally)
    - [Requirements](#requirements)
    - [Getting Lidar Data](#getting-lidar-data)
    - [Ouster YOLOv5 Demo with OpenCV](#ouster-yolov5-demo-with-opencv)
    - [Ouster YOLOv5 Demo with SimpleViz](#ouster-yolov5-demo-with-simplviz)
## Running the demo on Google colab
Follow the link ![Ouster_Yolo5_Demo](./Ouster_Yolo5_Demo.ipynb) to run the demo using Google Colab

## Running the demo locally

### Requirements
TODO: Complete the list of requirements
```bash
pip install -r requirements.txt
```

### Getting Lidar Data
The repo [yolov5-ouster-lidar-data](https://github.com/ouster-lidar/yolov5-ouster-lidar-data) contains some sample lidar data that you could use with the provided python examples. The two files `Ouster-YOLOv5-sample.json` and `Ouster-YOLOv5-sample.pcap` are currently expected to sit next to python scripts. 

### Ouster YOLOv5 Demo with OpenCV
![Ouster YOLOv5 Demo](./yolo5_opencv.py)

### Ouster YOLOv5 Demo with SimplViz
![Ouster YOLOv5 Demo / SimpleViz](./yolo5_simpleviz.py)