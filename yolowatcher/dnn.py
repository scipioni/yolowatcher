import os
import sys
import time

import cv2
import numpy as np

targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL,
           cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(net, frame, outs, confidence_min=0.5, classes=[], crop=True, step=1, save_dataset="",
                save_bbox=False, show_confidence=True, totals={'+': 0, '-': 0}, ratio=1.0, non_maximum_threshold=0.4):
    frameHeight, frameWidth = frame.shape[:2]

    # lato del quadrato massimo centrato
    Qh = min(frameHeight, frameWidth) if crop else frameHeight
    # lato del quadrato massimo centrato
    Qw = min(frameHeight, frameWidth) if crop else frameWidth
    deltax = int((frameWidth-Qw)/2) if crop else 0
    deltay = int((frameHeight-Qh)/2) if crop else 0

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence >= confidence_min:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    # Skip background label
                    classIds.append(int(detection[1]) - 1)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'DetectionOutput':  # SSD
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence >= confidence_min:
                    left = deltax + int(detection[3] * Qw)
                    top = deltay + int(detection[4] * Qh)
                    right = deltax + int(detection[5] * Qw)
                    bottom = deltay + int(detection[6] * Qh)
                    width = right - left + 1
                    height = bottom - top + 1
                    # Skip background label
                    classIds.append(int(detection[1]) - 1)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':  # YOLO
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        scales = []
        output_blob = 0
        for scale, out in enumerate(outs):  # ho 2 blob
            output_blob += 1
            for detection in out:  # detection 1x6
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence >= confidence_min:
                    center_x = int(detection[0] * Qw)
                    center_y = int(detection[1] * Qh)
                    width = int(detection[2] * Qw / ratio)
                    height = int(detection[3] * Qh * ratio)
                    left = deltax + center_x - width / 2
                    top = deltay + center_y - height / 2
                    probability_that_box_contain_object = detection[4]
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    scales.append(scale)
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_min, non_maximum_threshold)

    result = []
    for i in indices:
        i = i[0]
        confidence = confidences[i]
        if confidence > 0.8:
            if 'y+' in totals:
                totals['y+'] += 1
        elif confidence > 0.1:
            if 'y-' in totals:
                totals['y-'] += 1
        else:
            if 'y-bads' in totals:
                # mettiamo in bads anche le classi riconosciute male < 0.1
                totals['y-bads'] += 1
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        name = classIds[i]
        try:
            name = classes[name]
        except:
            pass

        result.append({'classId': classIds[i], 'confidence': confidences[i], 'box': (left, top, width, height),
                       'name': name, 'scale': scales[i]})
    if len(indices) == 0 and 'y-bads' in totals:  # non ho riconosciuto nulla
        totals['y-bads'] += 1
    totals['y-tot'] += len(indices)

    if result and len(result) > 1:
        # ordiniamo i caratteri secondo la formula x+y*1.5 così teniamo conto delle targhe quadrate con 2 righe
        result.sort(key=lambda x: x['box'][0] + 2.0*x['box'][1])

    if crop:
        cv2.rectangle(frame, (deltax, deltay),
                      (deltax+Qw, deltay+Qh), (255, 255, 0), 2)

    return result
