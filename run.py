import imghdr
import os
import glob
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
from load_model import human_dog_detector
from keras import backend
import tensorflow as tf

global graph
graph = tf.get_default_graph()


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'

def validate_image(stream):
    '''validates if a byte stream is one of an image'''
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')


@app.route('/')
def index():
    '''Index route for app'''
    return render_template('index.html')

@app.route('/result')
def result():
    '''Result route for app'''
    files = os.listdir(app.config['UPLOAD_PATH'])
    if files == []:
        return render_template('error.html', error='Error. You must sumbit a file first!')
    print(app.config['UPLOAD_PATH'] + '/' + files[0])
    with graph.as_default():
        message = human_dog_detector(app.config['UPLOAD_PATH'] + '/' + files[0])
    print(message)
    if "error" in message:
        return render_template('error.html', error=message)
    return render_template('result.html', files=files[0], message=message)

@app.route('/', methods=['POST'])
def upload_files():
    # delete past uploads
    files = glob.glob(app.config['UPLOAD_PATH'] + '/*')
    if files != []:
        for f in files:
            os.remove(f)
    #save upload
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('result'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True, threaded=False)


if __name__ == '__main__':
    main()