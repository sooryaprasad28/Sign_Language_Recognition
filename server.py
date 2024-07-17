from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import tensorflow as tf 

app = Flask(__name__)

# Load your sign language recognition model
model_path = 'logs/output_graph.pb'
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("logs/output_labels.txt")]

def load_model():
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

load_model()

def predict(image_data):
    with tf.compat.v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
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
sequence = []
def process_image(image_data):
    image_data = base64.b64decode(image_data)
    

    prediction, score = predict(image_data)
    sequence.append(prediction)
    sequence_str = ' '.join(sequence)
    
    return prediction, sequence

# Route to handle image recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data.get('image', '')

    if image_data:
        prediction, sequence = process_image(image_data)
        return jsonify({'prediction': prediction, 'sequence': sequence})
    else:
        return jsonify({'error': 'Invalid image data'})

if __name__ == '__main__':
    app.run(debug=True)
