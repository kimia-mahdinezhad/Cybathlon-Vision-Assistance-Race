import cv2
import zmq
import base64
import numpy as np,time
import pyshine as ps
from task7 import ImageProcessor

# www.pyshine.com
context = zmq.Context()
client_socket = context.socket(zmq.SUB)
client_socket.connect("tcp://192.168.144.90:5555")
client_socket.setsockopt_string(zmq.SUBSCRIBE,optval='')

image_processor = ImageProcessor()

while True:
    frame = client_socket.recv()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv2.imdecode(npimg, 1)

    image_processor.process_video(source)

    time.sleep(0.01)
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
cv2.destroyAllWindows()



