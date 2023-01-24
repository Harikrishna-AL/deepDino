import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
onnx_model = onnx.load("deepDino.onnx")
onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))

# x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('deepDino.onnx')
def run_model(model, x,y,classes):

    outputs = model.run(None, {'image': x})

    # Print Result 
    print(outputs[0][0])
    predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

    return predicted, actual
# x = Image.open('./data/screenshot170UP.png')
# x = x.resize((100,100))
# x = np.array(x)
# x = x.astype(np.float32)
# x = x.reshape(1,3,100,100)

img = cv2.imread('./data/screenshot170UP.png')
img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_AREA)
cv2.imwrite('greyscale.png', img)
img.resize((1, 1, 100, 100))

data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
pred,actual = run_model(ort_sess, data, 0,["UP", "DOWN", "RIGHT"])