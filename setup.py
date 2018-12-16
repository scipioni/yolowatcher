from setuptools import setup

setup(name='yolowatcher',
      version='0.1',
      description='yolo object detection as service',
      url='https://github.com/scipioni/yolowatcher',
      author='Stefano Scipioni',
      author_email='scipio.it@gmail.com',
      license='MIT',
      packages=['yolowatcher'],
      zip_safe=False,
      install_requires=['opencv-python', 'aionotify'],
      entry_points={
          'console_scripts': [
            'yolowatcher_download = yolowatcher.download:run',
            'yolowatcher_run = yolowatcher.main:run',
            ],
      })
