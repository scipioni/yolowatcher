# -*- coding: utf-8 -*-

# TODO https://www.home-assistant.io/components/camera.push/
# curl -X POST -H "x-ha-access: $PASS" -F "image=@/home/hass/camera/gate.jpg" http://localhost:8123/api/webhook/gate_webhook

import asyncio
import argparse
import aionotify
import os
from yolowatcher import detect


loop = asyncio.get_event_loop()

async def watch(args):
    watcher = aionotify.Watcher()
    watcher.watch(alias='incoming', path=args.folder, flags=aionotify.Flags.CLOSE_WRITE)

    await watcher.setup(loop)
    while True:
        event = await watcher.get_event()
        filename = os.path.join(args.folder, event.name)
        print(f"detected {filename}")

        bboxes = detect.detect(filename)
        for box in bboxes:
            print(box)
    watcher.close()

def run():
    args = detect.initialize()

    print(f"watching folder: {args.folder} ...")
    loop.run_until_complete(watch(args))
