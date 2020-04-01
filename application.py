from flask import Flask, render_template, url_for, request, redirect,send_file, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api,Resource
import os
from sqlalchemy import select, func
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from IPython.display import Image, display

app = Flask(__name__)
api = Api(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = load_model('model_weights.h5')
#CNN_model = load_model('CNN_model.h5')

def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im

def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred

    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

class Predict(Resource):
    def post(self):

        retJson ={}
        try:
            resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

            img = request.files.get('Image', '')

            test_img = get_encoding(resnet, img)

            Argmax_Search = predict_captions(test_img)

            z = Image(filename=img)

            display(z)

            print(Argmax_Search)

            retJson = {"status":200,"msg":"Successfully Predicted","caption":Argmax_Search}
        
        except Exception as e:
            print(e)
            retJson = {"status":400,"msg":"Unsuccessfull","caption":"Fail to predict"}

        return retJson


api.add_resource(Predict,"/predict")

if __name__ == "__main__":
    app.run(debug=True)