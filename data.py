# import pyautogui
import keyboard
import pyscreenshot as ImageGrab
from deepDino import run_model
import cv2
import numpy as np
import json
import onnx
import onnxruntime as ort

i=0
is_exit = False
def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey('esc', exit)
def data_collection():
    while (True):
        if is_exit:
            break
        if keyboard.is_pressed(keyboard.KEY_UP):

            ss = ImageGrab.grab(bbox=(625, 275, 775, 380))
            ss.save(f'./data/screenshot{i}'+ 'UP'+'.png')
            i+=1

        if keyboard.is_pressed(keyboard.KEY_DOWN):
            ss = ImageGrab.grab(bbox=(625, 275, 775, 380))
            ss.save(f'./data/screenshot{i}'+ 'DOWN'+'.png')
            i+=1

        if keyboard.is_pressed('right'):
            ss = ImageGrab.grab(bbox=(625, 275, 775, 380))
            ss.save(f'./data/screenshot{i}'+ 'RIGHT'+'.png')
            i+=1

def AIDino():
    deepDino = ort.InferenceSession('deepDino.onnx')
    for i in range(1):
    # while(1):
        if is_exit:
            break
        ss = ImageGrab.grab(bbox=(725, 250, 875, 355))
        ss.save('screenshot.png')
        img = np.array(ss)
        # img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_AREA)
        img.resize((1, 1, 100, 100))
        img = img/255
        data = json.dumps({'data': img.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        pred,actual = run_model(deepDino, data, 0,["UP", "DOWN", "RIGHT"])
        if pred == "UP":
            keyboard.press_and_release('up')
        if pred == "DOWN":
            keyboard.press_and_release('down')
        if pred == "RIGHT":
            keyboard.press_and_release('right')

if __name__ == "__main__":
    AIDino()