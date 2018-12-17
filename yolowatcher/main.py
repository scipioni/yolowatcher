# -*- coding: utf-8 -*-

# TODO https://www.home-assistant.io/components/camera.push/
# curl -X POST -H "x-ha-access: $PASS" -F "image=@/home/hass/camera/gate.jpg" http://localhost:8123/api/webhook/gate_webhook

import asyncio
import argparse
import os
from yolowatcher import detect
from subprocess import PIPE, Popen

loop = asyncio.get_event_loop()

inotifywait = ['inotifywait',
               '--recursive',
               '--quiet',
               '--monitor', ## '--timeout', '1',
               '--event',
               'CLOSE_WRITE',
               '--format', '%w%f']

async def watch(folder):
    p = Popen(inotifywait + [folder], stdout=PIPE)
    for line in iter(p.stdout.readline, ''):
        filename = line.strip().decode('utf8')
        print(f"detected {filename}")
        bboxes = detect.detect(filename)
        for box in bboxes:
            print(box)

def run():
    args = detect.initialize()

    folder = os.path.abspath(args.folder)

    print(f"watching folder: {folder} ...")
    loop.run_until_complete(watch(folder))
