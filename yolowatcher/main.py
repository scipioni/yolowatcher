# -*- coding: utf-8 -*-

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
