
def run_model(model, x,y,classes):

    outputs = model.run(None, {'image': x})
    predicted = classes[outputs[0][0].argmax(0)]
    print(f'Predicted: "{predicted}"')
    return predicted

