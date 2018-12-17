#!/bin/sh

python yolowatcher/detect.py \
	--yolo-model ~/.yolo/yolov3-tiny.weights \
	--yolo-config ~/.yolo/yolov3-tiny.cfg \
	--classes ~/.yolo/coco.names \
	$*
