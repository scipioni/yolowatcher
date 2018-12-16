YOLO object detector as service. It looks into folder for incoming images, detect object and call external RESTFUL API.

I use with homeassistant to detect peoples or cars on my cameras.

Download weigths and yolo configuration (yolov3-tiny as example)
```
yolowatcher_download --storage ./yolo
```


Test detect mode on sample file
```
python yolowatcher/detect.py examples/dog.jpg
```
