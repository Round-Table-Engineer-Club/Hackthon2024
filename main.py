import pickle

import cv2  # Import the OpenCV library for video capture and processing
import numpy as np
from flask import (
    Flask,
    Response,
    render_template,
)  # Import Flask for building a web application
import mediapipe as mp
from collections import Counter
from tokenizer.tokenizer import open_ai_formatting
from tokenizer.util import handTracker


# init
app = Flask(__name__)  # Create a Flask web application instance
camera = cv2.VideoCapture(0)  # Initialize a video capture object for camera index 1
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the frame width to 640 pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the frame height to 480 pixels

# model
tracker = handTracker()
model_path = "./hand/weight/best.pth"  # Update to your model file path

model_dict = pickle.load(
    open(
        "./hand/weight/model.p",
        "rb",
    )
)
model = model_dict["model"]


# init style
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}


def generate():
    global machine_output, cloud_text
    total_prediction = []
    counter = 0
    pred_buffer = []
    while True:
        key = cv2.waitKey(10)  # Wait 1 millisecond to check if a key has been pressed
        if key == 27:
            break

        # data
        data_aux = []
        x_ = []
        y_ = []

        success, frame = camera.read()  # Capture a frame from the camera
        if not success:  # If capturing a frame fails, exit the loop
            break

        # hand marking
        frame = tracker.handsFinder(frame)
        lmList = tracker.positionFinder(frame)
        results = tracker.results
        H, W, _ = frame.shape

        if results.multi_hand_landmarks:
            counter += 1

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if len(data_aux) <= 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    probe = model.predict_proba([np.asarray(data_aux)])[0][
                        int(prediction[0])
                    ]
                    predicted_character = labels_dict[int(prediction[0])]
                    if probe > 0.3:
                        pred_buffer.append(predicted_character)
                if counter >= 15 and len(pred_buffer) != 0:
                    counter = 0
                    machine_output += Counter(pred_buffer).most_common(1)[0][0]
                    print(Counter(pred_buffer).most_common(1)[0][0])

        elif len(machine_output) != 0:
            cloud_text = open_ai_formatting(machine_output)
            machine_output = ""

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

machine_output = ""


@app.route("/get-text")
def get_text():
    # put the logic here

    global test, machine_output
    return machine_output
    # print("Current test value:", test)
    # if test:
    #     test = False
    #     return "GGGOBUCKSSS"
    # else:
    #     test = True
    #     return "GGGOBUCKSSS"


cloudtest = True  # delete it after tput the logic

cloud_text = ""


@app.route("/get-cloud-text")
def get_cloud_text():
    # put the logic here

    global cloudtest, cloud_text

    return cloud_text
    # print("Current test value:", cloudtest)
    # if cloudtest:
    #     cloudtest = False
    #     return "GOBUCKS"
    # else:
    #     cloudtest = True
    #     return "GOBUCKS"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # Start the Flask web application with debugging enabled, listening on all available network interfaces and port 5000
