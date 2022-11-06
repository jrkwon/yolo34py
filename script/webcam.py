import time
import os
import argparse
import numpy as np
import cv2

import pydarknet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=int, 
                        help='video camera id number', default=0)
    parser.add_argument('-w', '--weight', type=str, 
                        help='weights: regular, tiny', default='regular')
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                        help ="minimum probability to filter weak detections")
    parser.add_argument("-y", "--yolo", default='../yolo', 
                        help="base path to YOLO directory")


    return parser.parse_args()


def main(args):
    
    # load the COCO class labels our YOLO model was trained on
    labels_path = os.path.sep.join([args.yolo, 'data/coco.names'])
    print(labels_path)
    LABELS = open(labels_path).read().strip().split("\n")
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)

    if args.weight == 'regular':
        weights_path = os.path.sep.join([args.yolo, "weights/yolov3.weights"])
        config_path = os.path.sep.join([args.yolo, "cfg/yolov3.cfg"])        
    else: # tiny
        weights_path = os.path.sep.join([args.yolo, "weights/yolov3-tiny.weights"])
        config_path = os.path.sep.join([args.yolo, "cfg/yolov3-tiny.cfg"])        
    
    coco_data_path = os.path.sep.join([args.yolo, "cfg/coco.data"])  
        
    net = pydarknet.Detector(bytes(config_path, encoding="utf-8"), 
                             bytes(weights_path, encoding="utf-8"), 
                             0, 
                             bytes(coco_data_path, encoding="utf-8"))

        
    # -------------------------------------------------------------------------
    # logo
    logo = cv2.imread('bimi_m_200x40.png')
    scale = 1
    size = (int(200*scale), int(40*scale))
    logo = cv2.resize(logo, size)
    padding = 20
    # Create a mask of logo
    img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

    cap = cv2.VideoCapture(args.video)

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = pydarknet.Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            # Frames per second can be calculated as 1 frame divided by time 
            # required to process 1 frame
            fps = 1 / (end_time - start_time)
            
            print(f'[INFO] FPS: {fps:2.4f}, Elapsed Time: {end_time-start_time:.4f}')

            for cat, score, bounds in results:
                if score < args.confidence:
                    continue
                
                cx, cy, w, h = bounds
                x, y = int(cx - w/2), int(cy - h/2)
                width, height = int(w), int(h)

    			# draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[LABELS.index(cat)]]

                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cat_score = f'{cat}: {score:.4f}'
                cv2.putText(frame, cat_score, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # -----------------------------------------------------------------
            # Logo 
            # Region of Image (ROI), where we want to insert logo
            roi = frame[-size[1]-padding:-padding, -size[0]-padding:-padding]
            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

            cv2.imshow("Video", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
    cap.release()

    
if __name__ == "__main__":
    args = parse_args()
    main(args)
