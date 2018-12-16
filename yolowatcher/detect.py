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

from yolowatcher.dnn import getOutputsNames, postprocess, targets

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
    (h, w) = image.shape[:2]

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
    global net, totals, args

    args = ap.parse_args()

    if args.classes:
        classes = [c.strip() for c in open(args.classes).readlines()]

    totals = {'+': 0, '-': 0, 'y+': 0, 'y-': 0, 'y-bads': 0, 'y-tot': 0,
              'y-inference-ms': 0, 'y-inference-count': 0, 'y-score': 0, 'y-truth': 0}

    try:
        net = cv2.dnn.readNetFromDarknet(args.yolo_config, args.yolo_model)
        print(f"yolo net initialized from {args.yolo_config} and {args.yolo_model}")
    except:
        print("Errore:", args.yolo_config, args.yolo_model)
        sys.exit(1)
    net.setPreferableTarget(args.target)
    return args


def detect(filename):
    global net, totals

    try:
        frame = cv2.imread(filename)
    except:
        print(" skipped")
        return []
        
    frame, bboxes = process_image(frame,
                                  net,
                                  totals,
                                  crop=args.crop,
                                  filename=filename,
                                  )
    return bboxes


if __name__ == '__main__':
    ap.add_argument('images', nargs='+', help="list of images")

    initialize()
    for image in args.images:
        bboxes = detect(image)
        for box in bboxes:
            print(box)
