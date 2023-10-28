import cv2  # Import the OpenCV library for video capture and processing
from flask import (
    Flask,
    Response,
    render_template,
)  # Import Flask for building a web application

app = Flask(__name__)  # Create a Flask web application instance
camera = cv2.VideoCapture(0)  # Initialize a video capture object for camera index 1
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the frame width to 640 pixels
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the frame height to 480 pixels


def generate():
    while True:
        success, frame = camera.read()  # Capture a frame from the camera
        if not success:  # If capturing a frame fails, exit the loop
            break
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
        return "z"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # Start the Flask web application with debugging enabled, listening on all available network interfaces and port 5000
