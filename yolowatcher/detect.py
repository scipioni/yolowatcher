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

from dnn import (drawRect, get_score, getNetImage,
                       getOutputsNames, postprocess, targets)
#from utils.utils import auto_canny, histogram_eq, rotate

ap = argparse.ArgumentParser()
ap.add_argument('images', nargs='+', help="list of images")
ap.add_argument('--yolo', action='store_true', default=True)
ap.add_argument("--yolo-model", default="weights/yolov3-tiny.weights")
ap.add_argument("--yolo-config", default="cfg/yolov3-tiny.cfg", help="dnn config network")
ap.add_argument('--yolo-size', type=int, default=416, help="yolo width/height size")
ap.add_argument("--confidence", type=float, default=0.3, help="minimum probability to filter weak detections")
#ap.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
ap.add_argument("--rotate", type=float, default=0.0)
ap.add_argument('--show', action='store_true', default=False)
ap.add_argument('--crop', action='store_true', default=False)
ap.add_argument('--save-bbox', default="")
ap.add_argument('--save-bbox-truth', default="")
ap.add_argument('--specials', default="./specials")
ap.add_argument('--flow', action='store_true', default=False)
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
ap.add_argument('--classes', default="cfg/coco.names", help="path to model.names")
args = ap.parse_args()


#tracker = cv2.TrackerKCF_create()

classes = ['plate']
if args.classes:
    classes = [c.strip() for c in open(args.classes).readlines()]

def process_image(image, net, totals, crop=False, filename='', bboxes_truth=[]):
    """
    se crop==True viene utilizzato solo il mirino quadrato centrato
    se crop==False viene fatto il padding dell'immagine fino ad avere un quadrato più grande che la contiene
    """
    (h, w) = image.shape[:2]
    
    #if not crop:
    #    lsquare = max(h, w)
        #image = cv2.copyMakeBorder(image, 0, lsquare-h, 0, lsquare-w, cv2.BORDER_CONSTANT, value=(255,255,255))
        #image = cv2.resize(image, (args.size, args.size), interpolation = cv2.INTER_LINEAR)
    #    (h, w) = image.shape[:2]

    if h < args.yolo_size or w < args.yolo_size:
        print(" resized")
        image = cv2.resize(image, (args.yolo_size, args.yolo_size), interpolation = cv2.INTER_LINEAR)
        (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (args.yolo_size, args.yolo_size), (0, 0, 0), swapRB=True, crop=crop)
    net.setInput(blob)
    detections = net.forward(getOutputsNames(net)) # ho 2 outputs 'yolo_13' e 'yolo_20'
    t, _ = net.getPerfProfile()
    inference_ms = t * 1000.0 / cv2.getTickFrequency()
    totals['y-inference-count'] += 1
    totals['y-inference-ms'] = inference_ms/totals['y-inference-count'] + totals['y-inference-ms']*(float(totals['y-inference-count']-1)/totals['y-inference-count'])
        
    bboxes = postprocess(net, image, detections, classes=classes, crop=crop, step=args.step, totals=totals)
    #score = np.mean([b['confidence'] for b in bboxes] or [0])*100.0/(abs(len(bboxes)-1)+1)
    score = get_score(bboxes_truth, bboxes)*100.0
    totals['y-score'] = score/totals['y-inference-count'] + totals['y-score']*(float(totals['y-inference-count']-1)/totals['y-inference-count'])
    if args.show:
        print("plate: scale1=%d scale2=%d score=%.1f" % (
            len([bbox['scale'] for bbox in bboxes if bbox['scale']==0]),
            len([bbox['scale'] for bbox in bboxes if bbox['scale']==1]),
            score,))
    return image, bboxes, score

class ImageGenerator():
    def __init__(self, images):
        self.video = None
        if images and '.mp4' in images[0]:
            self.video = images[0]
        self.images = images
        self.current = -1

        if self.video:
            print("video detected")
            if args.save_bbox or args.save_dataset:
                args.step = 4
                print("force step=%s" % args.step)
            self.cap = cv2.VideoCapture(self.video)
        else:
            self.cap = None

        self.current_frame = None
        self.current_bboxes = []

    def get(self, i):
        while self.current <  i:
            if self.video:
                ret, img = self.cap.read()
                if img is None:
                    return (None, '')
                self.current += 1
                self.current_frame = img
            elif 0 <= i < len(self.images):
                self.current = i
                self.current_frame = cv2.imread(self.images[i])
                self.current_bboxes = []
            else:
                return (None, '', [])
        if args.grey:
            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        if self.video:
            return self.current_frame.copy(), "%s-%d.jpg" % (os.path.basename(self.video).split('.')[0], i), []
        else:
            return self.current_frame.copy(), self.images[i], self.current_bboxes

def main():
    totals={'+':0, '-':0, 'y+':0, 'y-':0, 'y-bads':0, 'y-tot':0, 'y-inference-ms':0, 'y-inference-count':0, 'y-score':0, 'y-truth':0}
    try:
        net = cv2.dnn.readNetFromDarknet(args.yolo_config, args.yolo_model)
    except:
        print("Errore:", args.yolo_config, args.yolo_model)
        sys.exit(1)
    net.setPreferableTarget(args.target)
   
    if args.crop:
        pass #print("CROP active")

    imageGenerator = ImageGenerator(args.images)
    
    i = 0
    while True:
        frame, filename, bboxes_truth = imageGenerator.get(i)
        if frame is None:
            break
        
        frame, bboxes, score = process_image(frame, 
            net, 
            totals,
            crop=args.crop,
            filename=filename,
            bboxes_truth=bboxes_truth
            )

         
        if bboxes:
            print("+", end="")
            print(bboxes)
        else:
            print(".", end="")
        sys.stdout.flush()

        i += args.step


if __name__ == '__main__':
    main()
