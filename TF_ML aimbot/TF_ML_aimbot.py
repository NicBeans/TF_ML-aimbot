

import torch
import mss
import numpy as np
import cv2
import time
import keyboard 

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, pretrained=True)
with mss.mss() as sct:
    #capture dimensions
    monitor = {'top':20, 'left':0, 'width':1152, 'height':864}

while True:
    t = time.time()

    #grab screen image
    img = np.array(sct.grab(monitor))

    #model inference
    results = model(img)

    #display image, you need squeeze for this
    cv2.imshow('s', np.squeeze(results.render()))

    #get fps
    print('fps: {}'.format(1 / (time.time() - t)))

    #wait 1 ms
    cv2.waitKey(1)

    #break on p
    if keyboard.is_pressed('p'):
        break

cv2.destroyAllWindows()

