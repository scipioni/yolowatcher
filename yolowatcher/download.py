# -*- coding: utf-8 -*-

import argparse
import os
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--storage", default="yolo")
args = ap.parse_args()

def download(url, filename):
    abspath = os.path.join(args.storage, filename)
    if not os.path.exists(abspath):
        print(f"Download {url}")
        urllib.request.urlretrieve(url, abspath)
    else:
        print(f"{filename} is OK")

def run():
    if not os.path.exists(args.storage):
        print(f"Creating {args.storage}")
        os.makedirs(args.storage)

    download("https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg", "yolov3-tiny.cfg")
    download("https://pjreddie.com/media/files/yolov3-tiny.weights", "yolov3-tiny.weights")
    download("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", "coco.names")

