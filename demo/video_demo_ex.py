import time
import argparse
import numpy as np
import cv2

import pydarknet


def parse_args():
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('-i', '--input', metavar='input_video_path', type=str,
                        help='Path to source video', required=True)
    parser.add_argument('-o', '--output', metavar='output_video_path', type=str,
                        help='Path to destination video')
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                        help ="Minimum probability to filter weak detections")
    parser.add_argument('-w', '--weight', type=str, 
                        help='weights: regular, tiny', default='regular')
    parser.add_argument('-n', '--noshow', action = 'store_true',
                        help='Not showing video', default=False)
    
    return parser.parse_args()



def main(args):

    # load the COCO class labels our YOLO model was trained on
    labelsPath = 'data/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    print("[INFO] Source Path:", args.input)
    print("[INFO] Destination Path:", args.output)
    cap = cv2.VideoCapture(args.input)

    #########################################
    # logo
    # Read logo and resize
    logo = cv2.imread('bimi_m_200x40.png')
    scale = 1
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
    if args.output is not False:
        writer = cv2.VideoWriter(args.output,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)

    average_time = 0

    if args.weight == 'regular':
        net = pydarknet.Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), 
                                 bytes("weights/yolov3.weights", encoding="utf-8"), 
                                 0,
                                 bytes("cfg/coco.data", encoding="utf-8"))
    else: # tiny
        net = pydarknet.Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"), 
                                 bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 
                                 0,
                                 bytes("cfg/coco.data", encoding="utf-8"))

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = pydarknet.Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time-start_time) * 0.2
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)

            print(f"[INFO] FPS: {fps:2.4f}, Total Time: {end_time-start_time:.4f}: {average_time:.4f}")

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
            # logo
            # Create a mask of logo
            img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            roi = frame[-logo_size[1]-MARGIN_Y:-MARGIN_Y, -logo_size[0]-MARGIN_X:-MARGIN_X]
            # Set an index of where the mask is
            roi[np.where(mask)] = 0
            roi += logo

            if writer: 
                writer.write(frame)

            if not args.noshow:
                cv2.imshow("Video", frame)

        else:
            break

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break

    if writer: 
        writer.release()
    cap.release()


if __name__ == "__main__":
    main(parse_args())
