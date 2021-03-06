# -*- coding: utf-8 -*-

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from yolowatcher.dnn import getOutputsNames, postprocess, targets, drawRect

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="incoming")
ap.add_argument("--yolo-model", default="yolo/yolov3-tiny.weights")
ap.add_argument("--yolo-config", default="yolo/yolov3-tiny.cfg",
                help="dnn config network")
ap.add_argument('--yolo-size', type=int, default=416,
                help="yolo width/height size")
ap.add_argument("--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
ap.add_argument('--show', action='store_true', default=False)
ap.add_argument('--crop', action='store_true', default=False)
ap.add_argument('--square', action='store_true', default=False)
ap.add_argument('--grey', action='store_true', default=False)
ap.add_argument('--min-size', type=int, default=200)
ap.add_argument('--step', type=int, default=1)
ap.add_argument('--video-skip', type=int, default=10)
ap.add_argument('--target', choices=targets, default=cv2.dnn.DNN_TARGET_CPU, type=int,
                help='Choose one of target computation devices: '
                '%d: CPU target (by default), '
                '%d: OpenCL, '
                '%d: OpenCL fp16 (half-float precision), '
                '%d: VPU' % targets)
ap.add_argument('--classes', default="yolo/coco.names",
                help="path to model.names")

classes = ['plate']
net = None
totals = {}


def process_image(image, net, totals, crop=False, filename='', bboxes_truth=[]):
    """
    se crop==True viene utilizzato solo il mirino quadrato centrato
    se crop==False viene fatto il padding dell'immagine fino ad avere un quadrato più grande che la contiene
    """
    try:
        (h, w) = image.shape[:2]
    except:
        return None, []

    if h < args.yolo_size or w < args.yolo_size:
        print(" resized")
        image = cv2.resize(
            image, (args.yolo_size, args.yolo_size), interpolation=cv2.INTER_LINEAR)
        (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 1.0/255.0, (args.yolo_size, args.yolo_size), (0, 0, 0), swapRB=True, crop=crop)
    net.setInput(blob)
    # ho 2 outputs 'yolo_13' e 'yolo_20'
    detections = net.forward(getOutputsNames(net))
    t, _ = net.getPerfProfile()
    inference_ms = t * 1000.0 / cv2.getTickFrequency()
    totals['y-inference-count'] += 1
    totals['y-inference-ms'] = inference_ms/totals['y-inference-count'] + \
        totals['y-inference-ms'] * \
        (float(totals['y-inference-count']-1)/totals['y-inference-count'])

    bboxes = postprocess(net, image, detections, classes=classes,
                         crop=crop, step=args.step, totals=totals)
    return image, bboxes


def initialize():
    global net, totals, args, classes

    args = ap.parse_args()

    if args.classes:
        classes = [c.strip() for c in open(args.classes).readlines()]

    totals = {'+': 0, '-': 0, 'y+': 0, 'y-': 0, 'y-bads': 0, 'y-tot': 0,
              'y-inference-ms': 0, 'y-inference-count': 0, 'y-score': 0, 'y-truth': 0}

    try:
        net = cv2.dnn.readNetFromDarknet(args.yolo_config, args.yolo_model)
        print(
            f"yolo net initialized from {args.yolo_config} and {args.yolo_model}")
    except:
        print("Errore:", args.yolo_config, args.yolo_model)
        sys.exit(1)
    net.setPreferableTarget(args.target)
    return args


def detect(filename):
    global net, totals

    isImage = '.jpg' in filename

    if not isImage:
        print(f"processing video {filename}")
        cap = cv2.VideoCapture(filename)
    else:
        print(f"processing image {filename}")

    i = 1
    while True:
        if isImage:
            try:
                frame = cv2.imread(filename)
            except:
                print(" skipped")
                frame = None
        else:
            while i % args.video_skip > 0:
                ret, frame = cap.read()
                i += 1

        if frame is None:
            break  # yield []

        frame, bboxes = process_image(frame,
                                      net,
                                      totals,
                                      crop=args.crop,
                                      filename=filename,
                                      )

        if frame is None:
            break
        if args.show:
            for box in bboxes:
                drawRect(frame, box, classes=classes)

            cv2.imshow("image", frame)
            key = cv2.waitKey(0)
            if key in (ord('q'), 27):
                sys.exit(0)
        if not bboxes:
            print(".", end="")
            sys.stdout.flush()
        yield bboxes

        if isImage:
            break
        i += 1


if __name__ == '__main__':
    ap.add_argument('images', nargs='+', help="list of images")

    initialize()
    for image in args.images:
        for bboxes in detect(image):
            for box in bboxes:
                print(box)
