import torch
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime
import serial

port = "COM7"
baud = 9600
ser = serial.Serial(port, baud, timeout=1)
number_of_samples = 0
MiddleX = [320, 320, 320, 320, 320]
MiddleY = [240, 240, 240, 240, 240]

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', required=False, default="0", help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=True, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=True, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')
    parser.add_argument('--cuda', action='store_true', default=True, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

    args = parser.parse_args()

    return args

def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches

def draw_bbox(imgs, bbox, colors, classes):
    global number_of_samples
    global MiddleX
    global MiddleY
    img = imgs[int(bbox[0])]
    label = classes[int(bbox[-1])]
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())
    #color_value = int(bbox[-1]) + 120
    #color = random.choice(colors)#(color_value, color_value, color_value)#random.choice(colors)#(255, 0, 0)
    color = colors[int(bbox[-1])]
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    type_of_object = int(bbox[-1])
    if type_of_object == 67: #64 if its a mouse, 0 is for a person, 67 for cell phone
        #find the middle point

        MiddleX[0] = MiddleX[1]
        MiddleX[1] = MiddleX[2]
        MiddleX[2] = MiddleX[3]
        MiddleX[3] = MiddleX[4]
        MiddleX[4] = int(((p2[0] - p1[0])/2) + p1[0])
        MiddleY[0] = MiddleY[1]
        MiddleY[1] = MiddleY[2]
        MiddleY[2] = MiddleY[3]
        MiddleY[3] = MiddleY[4]
        MiddleY[4] = int(((p2[1] - p1[1])/2) + p1[1])
        MidX = torch.from_numpy(np.array(int((MiddleX[4] + MiddleX[3] + MiddleX[2] + MiddleX[1] + MiddleX[0])/5), dtype='int32'))
        MidY = torch.from_numpy(np.array(int((MiddleY[4] + MiddleY[3] + MiddleY[2] + MiddleY[1] + MiddleY[0])/5), dtype='int32'))

        Box_MiddleP1 = (MidX - 5, MidY - 5)
        Box_MiddleP2 = (MidX + 5, MidY + 5)
        #Box_MiddleP1 = torch.from_numpy(np.array(Box_MiddleP1, dtype='int32'))
        #Box_MiddleP2 = torch.from_numpy(np.array(Box_MiddleP2, dtype='int32'))
        cv2.rectangle(img, Box_MiddleP1, Box_MiddleP2, color, 2)
        #slow down the move_camera process. The response is slow. Fix this
        if number_of_samples <= 1:
            number_of_samples += 1
        else:
            number_of_samples = 0
            #Values for 640x480. Move camera if object is not in the middle
            if (MidX < 280):
                move_camera("X", "+")
            elif (MidX > 360):
                move_camera("X","-")
            elif (MidY < 200):
                move_camera("Y", "+")
            elif (MidY > 280):
                move_camera("Y","-")

def move_camera(axis, position):
    #ser = serial.Serial(port, baud, timeout=1)
    if ser.isOpen():
        if (axis == "X"):
            if (position == "+"):
                ser.write(b'2,+' + bytes([10]))#bytes([13, 10]))
            elif (position == "-"):
                ser.write(b'2,-' + bytes([10]))#bytes([13, 10]))
        elif (axis == "Y"):
            if (position == "+"):
                ser.write(b'1,+' + bytes([10]))#bytes([13, 10]))
            elif (position == "-"):
                ser.write(b'1,-' + bytes([10]))#bytes([13, 10]))
    #ser.close()

def detect_video(model, args):

    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")
    #colors = [colors[1]]
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')
    while cap.isOpened():
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh)
            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)
                for detection in detections:
                    draw_bbox([frame], detection, colors, classes)

            if not args.no_show:
                cv2.imshow('frame', frame)
            out.write(frame)
            if read_frames % 30 == 0:
                print('Number of frames processed:', read_frames)
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    return


def detect_image(model, args):

    print('Loading input image(s)...')
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    batch_size = int(model.net_info['batch'])

    imlist, imgs = load_images(args.input)
    print('Input image(s) loaded')

    img_batches = create_batches(imgs, batch_size)

    # load colors and classes
    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")

    if not osp.exists(args.outdir):
        os.makedirs(args.outdir)

    start_time = datetime.now()
    print('Detecting...')
    for batchi, img_batch in enumerate(img_batches):
        img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
        img_tensors = torch.stack(img_tensors)
        img_tensors = Variable(img_tensors)
        if args.cuda:
            img_tensors = img_tensors.cuda()
        detections = model(img_tensors, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)
        if len(detections) == 0:
            continue

        detections = transform_result(detections, img_batch, input_size)
        for detection in detections:
            draw_bbox(img_batch, detection, colors, classes)

        for i, img in enumerate(img_batch):
            save_path = osp.join(args.outdir, 'det_' + osp.basename(imlist[batchi*batch_size + i]))
            cv2.imwrite(save_path, img)

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))

    return

def main():

    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    if args.video:
        detect_video(model, args)

    else:
        detect_image(model, args)



if __name__ == '__main__':
    main()