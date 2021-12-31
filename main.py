import io
import os
from pickle import load

from waitress import serve

import numpy as np
from PIL import Image
from flask import Flask, request, flash, jsonify
from keras.applications.xception import Xception
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            msg = 'No file part'
            flash(msg)
            return jsonify(msg), 400

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            msg = 'No selected file'
            flash(msg)
            return jsonify({
                'status': 'error',
                'message': msg
            }), 400

        if not file or not allowed_file(file.filename):
            msg = 'Allowed image types are -> jpg, jpeg'
            flash(msg)
            return jsonify({
                'status': 'error',
                'message': msg
            }), 400
        else:
            try:
                filename = secure_filename(file.filename)
                print('upload_image filename: ' + filename)
                msg = 'Image successfully uploaded'
                flash(msg)

                max_length = 32
                cur_dir = os.getcwd()
                tokenizer = load(open(os.path.join(cur_dir, 'notebooks', "tokenizer.p"), "rb"))
                model = load_model(os.path.join(cur_dir, 'models_in_use', "model_10.h5"))
                xception_model = Xception(include_top=False, pooling="avg")

                photo = extract_features(file.stream.read(), xception_model)

                description = generate_desc(model, tokenizer, photo, max_length)
                return jsonify({
                    'status': 'success',
                    'message': msg,
                    'description': description
                }), 200
            except Exception as e:
                print(e)
                return jsonify({
                    'status': 'error',
                    'message': 'Error while extracting features'
                }), 500

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def extract_features(image_stream, model):
    image = Image.open(io.BytesIO(image_stream))

    image = image.resize((299, 299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)

    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = ''
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        if word == 'end':
            break
        else:
            in_text += word + " "
    return in_text


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
