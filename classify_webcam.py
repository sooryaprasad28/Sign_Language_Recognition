# import sys
# import os
# import cv2
# import base64
# from io import BytesIO

# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# import copy

# # Disable tensorflow compilation warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
# from socketio import server
# from socketio import emit


# # Replace with your label file path
# label_lines = [line.rstrip() for line in open("logs/output_labels.txt")]

# # Unpersists graph from file
# with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
#     graph_def = tf.compat.v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     _ = tf.import_graph_def(graph_def, name='')

# with tf.compat.v1.Session() as sess:
#     # Feed the image_data as input to the graph and get first prediction
#     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

#     app = server()
#     app.wrap_http(sys.argv[1])  # Replace with your server instance

#     @app.event
#     def connect(sid, environ):
#         print('Client connected:', sid)

#     @app.event
#     def disconnect(sid):
#         print('Client disconnected:', sid)

#     def decode_frame_data(data):
#         # Decode base64 data to image bytes
#         image_data = base64.b64decode(data.split(',')[1])
#         # Convert bytes to NumPy array
#         image_array = np.frombuffer(image_data, dtype=np.uint8)
#         # Decode the array as a color image
#         return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

#     @app.event
#     def frame(sid, data):
#         # Process received frame data
#         image = decode_frame_data(data)

#         # Extract ROI or process the image as needed
#         x1, y1, x2, y2 = 100, 100, 300, 300  # Adjust ROI coordinates
#         img_cropped = image[y1:y2, x1:x2]

#         # Encode image as JPEG string
#         _, img_buffer = cv2.imencode('.jpg', img_cropped)
#         image_data = img_buffer.tobytes()

#         # Run prediction using your existing code
#         label, score = predict(image_data)

#         # Send prediction back to the frontend
#         emit('prediction', {'label': label, 'score': score}, sid)

#     @app.event
#     def clear(sid):
#         # Clear the text sequence (modify based on your implementation)
#         text_sequence = ''
#         emit('clear_text', text_sequence, sid)

#     if __name__ == '__main__':
#         app.run()




import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def predict(image_data):

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("logs/output_labels.txt")]

# Unpersists graph from file
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0

    cap = cv2.VideoCapture(0)

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        
        if ret:
            x1, y1, x2, y2 = 100, 100, 300, 300
            img_cropped = img[y1:y2, x1:x2]

            c += 1
            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
            
            a = cv2.waitKey(1) # waits to see if `esc` is pressed
            
            if i == 4:
                res_tmp, score = predict(image_data)
                res = res_tmp
                i = 0
                if mem == res:
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive == 2 and res not in ['nothing']:
                    if res == 'space':
                        sequence += ' '
                    elif res == 'del':
                        sequence = sequence[:-1]
                    else:
                        sequence += res
                    consecutive = 0
            i += 1
            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            mem = res
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.imshow("img", img)
            img_sequence = np.zeros((200,1200,3), np.uint8)
            cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow('sequence', img_sequence)
            
            if a == 27: # when `esc` is pressed
                break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()