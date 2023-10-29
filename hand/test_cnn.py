import torch
import torch.nn as nn
from torch.autograd import Variable
from train_cnn import CNN
from PIL import Image
import numpy as np

# Assuming the CNN class definition is the same as in your previous training script


# Load the pre-trained model
def load_model(device, model_path):
    print("using {} device.".format(device))
    # Initialize the model
    model = CNN().to(device)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path))

    # Setting it to evaluation mode, this is important as it turns off features like dropout
    model.eval()
    return model

def predict(image, model, device):
    '''
    Make a prediction for a single image.

    Args:
        image (Tensor): A tensor containing the image to predict, shape should be (1, 28, 28) or corresponding dimensions.
    
    Returns:
        prediction (int): The predicted class label.
    '''
    # If the image is not a tensor, conversion code might be needed here
    # For instance, if the image is a PIL image, you might need to use transforms to convert it to a tensor
    
    # Add an extra dimension (batch size = 1)
    image = image.unsqueeze(0)  # the shape of the image is now (1, 1, 28, 28) if the original image was (1, 28, 28)
    # Move the image to the device
    image = image.to(device)

    # Make a prediction using the model
    output = model(image)

    # Retrieve the highest probability class
    prediction = torch.argmax(output, dim=1)  # this will give us the results for each item in the batch

    return prediction.item()  # since batch size is 1, we just get the first item

# Now, you can call `predict(image)` to predict the label of a new image.
# Make sure `image` is a tensor of the correct shape, i.e., (1, 28, 28) or (28, 28) but would need unsqueeze.
# open C:\Users\xiaog\Documents\Code\Hackthon\hand\test_1.jpg
def main():
    model_path = 'weight/best.pth'  # Update to your model file path
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model=load_model(device,model_path)
    image = Image.open('C:/Users/54189/Documents/python_workspace/Hackthon2024/hand/test_1.jpg')
    image = image.convert('L')
    image = image.resize((28,28))
    image = np.array(image)
    image = torch.from_numpy(image).float() 
    image = image.unsqueeze(0)
    result=predict(image,model,device)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    predicted_char = alphabet[result]  

    print(f"Predicted character: {predicted_char}") 
    print(result)

if __name__ == "__main__":
    print("Running prediction script...")
    main()
