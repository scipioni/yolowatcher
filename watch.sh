#!/bin/sh

yolowatcher_run --yolo-model ~/.yolo/yolov3-tiny.weights --yolo-config ~/.yolo/yolov3-tiny.cfg --classes ~/.yolo/coco.names --folder incoming

