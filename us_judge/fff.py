import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.models import model_from_json
import json
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from keras import optimizers
from keras.applications.vgg16 import VGG16


classes = ["うどん","そば"]
num_classes = len(classes)
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENDSION = set(['png','jpg','jpeg'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENDSION

model = model_from_json(open('model.json').read())
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])
model.load_weights('model_weights.h5')

# model = load_model('./model_weights.h5',compile=False)

graph = tf.get_default_graph()

@app.route('/',methods=['GET','POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER,filename))
                filepath = os.path.join(UPLOAD_FOLDER,filename)

                img = image.load_img(filepath,target_size=(image_size,image_size))
                img = image.img_to_array(img)
                data = np.array([img])
                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_answer = "これは " + classes[predicted] + " です"

                return render_template("index.html",answer = pred_answer)
        return render_template("index.html",answer="")
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
        

