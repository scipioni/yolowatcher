YOLO object detector as service. It looks into folder for incoming images, detect object and call external RESTFUL API.

I use with homeassistant to detect peoples or cars on my cameras.

Download weigths and yolo configuration (yolov3-tiny as example)
```
yolowatcher_download --storage ~/.yolo
```

Test detect mode on sample file
```
python yolowatcher/detect.py --yolo-model ~/.yolo/yolov3-tiny.weights --yolo-config ~/.yolo/yolov3-tiny.cfg --classes ~/.yolo/coco.names examples/dog.jp
```

Run watcher to look in folder
```
yolowatcher_run --yolo-model ~/.yolo/yolov3-tiny.weights --yolo-config ~/.yolo/yolov3-tiny.cfg --classes ~/.yolo/coco.names --folder incoming
```

On another shell copy file on incoming
```
cp examples/dog.jpg incoming
```


