from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

time.sleep(2)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <html>
            <body>
                <h2>Raspberry Pi Live Stream</h2>
                <img src="/video_feed">
            </body>
        </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Live streaming started...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
