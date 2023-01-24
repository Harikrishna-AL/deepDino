import pyautogui
import keyboard
import pyscreenshot as ImageGrab
from deepDino import run_model
i=0
is_exit = False
def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey('esc', exit)
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