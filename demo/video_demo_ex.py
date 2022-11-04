import time
import argparse
import numpy as np
import cv2

from pydarknet import Detector, Image


# BGR format
BOX_COLOR = (0, 255, 255)
TEXT_COLOR = (0, 255, 128)

def parse_args():
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('-c', '--camera', type=int, help='camera id number', default=0)
    parser.add_argument('-i', '--input', metavar='input_video_path', type=str,
                        help='Path to source video')
    parser.add_argument('-o', '--output', metavar='output_video_path', type=str,
                        help='Path to destination video')

    return parser.parse_args()



def main(args):
    print("Source Path:", args.input)
    print("Destination Path:", args.output)
    cap = cv2.VideoCapture(args.input)

    #########################################
    # logo
    # Read logo and resize
    logo = cv2.imread('bimi_m_200x40.png')
    scale = 1.5
    MARGIN_X = MARGIN_Y = 15
    logo_size = (int(scale*200), int(scale*40))
    logo = cv2.resize(logo, logo_size)

    #########################################
    # video writer
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(args.output,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    average_time = 0

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time-start_time) * 0.2
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)

            print("FPS: ", fps)
            print("Total Time:", end_time-start_time, ":", average_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)), BOX_COLOR)
                cat_score = f'{cat}:{score:.2f}'
                cv2.putText(frame, cat_score, (int(x-w/2+5),int(y-h/2+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR)

            # logo
            # Create a mask of logo
            img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            roi = frame[-logo_size[1]-MARGIN_Y:-MARGIN_Y, -logo_size[0]-MARGIN_X:-MARGIN_X]
            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

            result.write(frame)
            cv2.imshow("preview", frame)

        else:
            break

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    main(parse_args())
