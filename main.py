import cv2  # Import the OpenCV library for video capture and processing
import numpy as np
import torch
from flask import (
    Flask,
    Response,
    render_template,
)  # Import Flask for building a web application
from torch import nn


from tokenizer.util import handTracker


# init
app = Flask(__name__)  # Create a Flask web application instance
camera = cv2.VideoCapture(0)  # Initialize a video capture object for camera index 1
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the frame width to 640 pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the frame height to 480 pixels



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,26)
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x


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




tracker = handTracker()
model_path = 'C:/Users/54189/Documents/python_workspace/Hackthon2024/hand/weight/best.pth'  # Update to your model file path
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = load_model(device, model_path)

def generate():

    while True:
        success, frame = camera.read()  # Capture a frame from the camera
        if not success:  # If capturing a frame fails, exit the loop
            break

        # hand marking
        frame = tracker.handsFinder(frame)
        lmList = tracker.positionFinder(frame)
        if len(lmList) != 0:
            print(lmList[4])

        # image = image.convert('L')
        # predict_image = cv2.resize(frame, (28, 28))
        # predict_image = np.array(predict_image)
        # predict_image = torch.from_numpy(predict_image).float()
        # predict_image = predict_image.unsqueeze(0)
        # result = predict(predict_image, model, device)
        # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # predicted_char = alphabet[result]
        #
        # print(f"Predicted character: {predicted_char}")
        # print(result)

        ret, buffer = cv2.imencode(".jpg", frame)  # Encode the frame as a JPEG image

        frame = buffer.tobytes()  # Convert the frame to bytes
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # Yield the frame as part of a multipart response
    camera.release()  # Release the camera object when done
    cv2.destroyAllWindows()  # Close any OpenCV windows


# @app.route('/')

# def index():
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
# Define the root route, which returns a multipart response with the video stream


@app.route("/")
def display_page():
    return render_template("index.html")


@app.route("/video_camera")
def video_camera():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


test = True  # delete it after tput the logic


@app.route("/get-text")
def get_text():
    # put the logic here

    global test
    print("Current test value:", test)
    if test:
        test = False
        return "abc"
    else:
        test = True
        return "abcc"


cloudtest = True  # delete it after tput the logic


@app.route("/get-cloud-text")
def get_cloud_text():
    # put the logic here

    global cloudtest
    print("Current test value:", cloudtest)
    if cloudtest:
        cloudtest = False
        return "efh"
    else:
        cloudtest = True
        return "efg"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # Start the Flask web application with debugging enabled, listening on all available network interfaces and port 5000
