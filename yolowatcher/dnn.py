import os
import sys
import time

import cv2
import numpy as np

global_counter = 0

targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)

def optimize_bbox(image, gridsize=16):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def save_image_bbox(image, path, min_size=100, min_ratio=30/25.0):
    path_bbox = os.path.join(path, "%s.jpg" % int(time.time()*1000))
    #if args.optimize_bbox:
    #    image = optimize_bbox(image)

    (hc, wc) = image.shape[:2]
    if wc < min_size:
        print("too small: %d < %d" % (wc, min_size))
        return image

    if hc > 0 and float(wc)/hc < min_ratio:
        print("bad ratio: %f" % (float(wc)/hc))
        return image

    # try:
    #     if wc >= hc:
    #         W = size
    #         H = int(hc*(float(size)/wc))
    #     else:
    #         W = int(hc*(float(size)/hc))
    #         H = size
    #     image_scaled = cv2.resize(image, (W, H), interpolation = cv2.INTER_LINEAR)
    #     print("\nresize from %dx%d to %dx%d: %s" % (wc, hc, W, H, path_bbox), end=' ')
    # except:
    #     print("bbox not saved")
    #     return image

    #if args.square: # pad
    #    (hc, wc) = image_scaled.shape[:2]
    #    lsquare = max(hc, wc)
    #    image_scaled = cv2.copyMakeBorder(image_scaled, 0, lsquare-hc, 0, lsquare-wc, cv2.BORDER_CONSTANT, value=(0,0,0))
    
    cv2.imwrite(path_bbox, image)
    return image

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess(net, frame, outs, confidence_min=0.5, classes=[], crop=True, step=1, save_dataset="", 
    save_bbox=False, show_confidence=True, totals={'+':0, '-':0}, ratio=1.0, non_maximum_threshold=0.4):
    global global_counter
    frameHeight, frameWidth = frame.shape[:2]

    Qh = min(frameHeight,frameWidth) if crop else frameHeight # lato del quadrato massimo centrato
    Qw = min(frameHeight,frameWidth) if crop else frameWidth # lato del quadrato massimo centrato
    deltax = int((frameWidth-Qw)/2) if crop else 0
    deltay = int((frameHeight-Qh)/2) if crop else 0

#    def drawPred(classId, conf, left, top, right, bottom, color=(0,255,0)):
#        # Draw a bounding box.
#        left = int(left)
#        top = int(top)
#        right = int(right)
#        bottom = int(bottom)
#        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#
#        if classes:
#            assert(classId < len(classes))
#            label = classes[classId]
#        else:
#            label = str(classId+1)
#        if show_confidence:
#            label += " %.1f%%" % (int(conf*1000)/10.0,)
#
#        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#        top = max(top, labelSize[1])
#        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (0, 255, 0), cv2.FILLED)
#        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

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
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'DetectionOutput': # SSD
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
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region': # YOLO
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        scales = []
        output_blob = 0
        for scale,out in enumerate(outs): # ho 2 blob
            output_blob += 1
            for detection in out: # detection 1x6
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_min, non_maximum_threshold)

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
                totals['y-bads'] += 1 # mettiamo in bads anche le classi riconosciute male < 0.1
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2])
        height = int(box[3])
        if save_bbox:
            frame[top:top+height, left:left+width] = save_image_bbox(frame[top:top+height, left:left+width], save_bbox)
        #drawPred(classIds[i], confidences[i], left, top, left + width, top + height, color=(255,0,0) if global_counter==1 else (0,255,0))
        name=classIds[i]
        try:
            name = classes[name]
        except:
            pass

        result.append({'classId':classIds[i], 'confidence':confidences[i], 'box':(left,top,width,height), 
            'name':name, 'scale':scales[i]})
    if len(indices) == 0 and 'y-bads' in totals: # non ho riconosciuto nulla
        totals['y-bads'] += 1
    totals['y-tot'] += len(indices)

    if result and len(result) > 1:
        # ordiniamo i caratteri secondo la formula x+y*1.5 così teniamo conto delle targhe quadrate con 2 righe
        result.sort(key=lambda x: x['box'][0] + 2.0*x['box'][1])
    
    if crop:
        cv2.rectangle(frame, (deltax, deltay), (deltax+Qw, deltay+Qh),(255, 255, 0), 2)

    return result

def getNetImage(image, crop=False, size=300):
    (h, w) = image.shape[:2]
    Qh=min(h,w) if crop else h # lato del quadrato massimo centrato
    Qw=min(h,w) if crop else w # lato del quadrato massimo centrato
    deltax = int((w-Qw)/2) if crop else 0
    deltay = int((h-Qh)/2) if crop else 0
    if crop:
        image = image[deltay:deltay+Qh,deltax:deltax+Qw]
    return cv2.resize(image, (size, size), interpolation = cv2.INTER_LINEAR)


def get_IOU_box(box1, box2):
    boxA = [box1['box'][0], box1['box'][1], box1['box'][0]+box1['box'][2], box1['box'][1]+box1['box'][3]]
    boxB = [box2['box'][0], box2['box'][1], box2['box'][0]+box2['box'][2], box2['box'][1]+box2['box'][3]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_score(boxes_truth, boxes2):
    if not boxes_truth or not boxes2:
        return 0.0
    scores = []
    for box1 in boxes_truth:
        ious_box1 = [get_IOU_box(box1, box2) for box2 in boxes2 if box2['classId']==box1['classId']]
        if not ious_box1:
            score = 0.0
        else:
            iou_max = max(ious_box1) 
            i = ious_box1.index(iou_max) # questo è l'elemento di boxes2 che corrisponde a box1
            score = ious_box1[i] * min(box1['confidence'], boxes2[i]['confidence'])
            boxes2[i]['score'] = score # aggiungiamo lo score
        scores.append(score)
    return sum(scores)/len(scores)


def drawRect(frame, box, classes=[], color=(0,255,0), show_confidence=True):
    # Draw a bounding box.
    left = box['box'][0]
    top = box['box'][1]
    right = box['box'][0] + box['box'][2]
    bottom = box['box'][1] + box['box'][3]
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    if classes:
        assert(box['classId'] < len(classes))
        label = classes[box['classId']]
    else:
        label = str(box['classId']+1)

    if show_confidence:
        if show_confidence:
            label += " %.1f%%" % (int(box['confidence']*1000)/10.0,)
            if 'score' in box:
                label += " s:%.1f%%" % (int(box['score']*1000)/10.0,)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
