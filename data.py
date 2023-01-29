
import keyboard
import pyscreenshot as ImageGrab
from deepDino import run_model
import numpy as np
import onnxruntime as ort

i=0
is_exit = False
def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey('esc', exit)
def data_collection():
    global i
    while (True):
        if is_exit:
            break
        if keyboard.is_pressed(keyboard.KEY_UP):

            ss = ImageGrab.grab(bbox=(660, 275, 785, 380))
            ss.save(f'./new_data/screenshot{i}'+ 'UP'+'.png')
            i+=1

        if keyboard.is_pressed(keyboard.KEY_DOWN):
            ss = ImageGrab.grab(bbox=(660, 275, 785, 380))
            ss.save(f'./new_data/screenshot{i}'+ 'DOWN'+'.png')
            i+=1

        if keyboard.is_pressed('right'):
            ss = ImageGrab.grab(bbox=(660, 275, 785, 380))
            ss.save(f'./new_data/screenshot{i}'+ 'RIGHT'+'.png')
            i+=1

def AIDino():
    deepDino = ort.InferenceSession('deepDino.onnx')
    while(1):
        if is_exit:
            break
        ss = ImageGrab.grab(bbox=(660, 275, 785, 380))
        ss = ss.convert('L')
        ss = ss.resize((100, 100))
        data = np.array(ss.getdata()).reshape(1,ss.size[0], ss.size[1])
        data = data/255
        data = data.reshape((1,1,100,100))
        data = data.astype(np.float32)
        pred = run_model(deepDino, data, 0,["UP","RIGHT"])
        if pred == "UP":
            keyboard.press_and_release('up')
        if pred == "RIGHT":
            keyboard.press_and_release('right')

def test():
    global i
    # while(1):
    for i in range(50):
        i+=1
        if is_exit:
            break
        ss = ImageGrab.grab(bbox=(660, 275, 785, 380))
        ss.save(f'screenshot{i}.png')

if __name__ == "__main__":
    AIDino()
    # test()
    # data_collection()