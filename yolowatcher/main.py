import asyncio

import aionotify

# Setup the watcher
watcher = aionotify.Watcher()
watcher.watch(alias='imagess', path='images',
              flags=aionotify.Flags.CLOSE_WRITE)

# Prepare the loop
loop = asyncio.get_event_loop()


async def work():
    await watcher.setup(loop)
    while True:
        event = await watcher.get_event()
        print(event)
    watcher.close()

loop.run_until_complete(work())
