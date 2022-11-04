import time
import argparse

import pydarknet
from pydarknet import Detector, Image
import cv2

# BGR format
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 255, 128)

def main(camera):
    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    cap = cv2.VideoCapture(camera)

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)
            
            print("FPS: ", fps)
            print("Elapsed Time:",end_time-start_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)), BOX_COLOR)
                #cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,255,0))
                cat_score = f'{cat}:{score:.2f}'
                cv2.putText(frame, cat_score, (int(x-w/2+5),int(y-h/2+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR)
                #cv2.putText(frame, cat, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
    cap.release()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', type=int, help='camera id number', default=0)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    main(args.camera)
