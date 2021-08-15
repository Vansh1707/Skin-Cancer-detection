import numpy as np
from flask import Flask,render_template,request
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import os
# from werkzeug import secure_filename
app=Flask(__name__)

UPLOAD_PATH='./static/images_stored'
model=load_model('mymodel5.h5')
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            img_location=os.path.join(UPLOAD_PATH,image_file.filename)
            image_file.save(img_location)
            img_pred=image.load_img(img_location,target_size=(150,150))
            x = image.img_to_array(img_pred)
            x=np.expand_dims(x,axis=0)
            images=np.vstack([x])
            val=model.predict(images)
            hu=np.argmax(val, axis =1)
            st=''
            if hu==0:
                st="Melanocytic nevi"
            elif hu==1:
                st="dermatofibroma"
            elif hu==2:
                st="Benign keratosis-like lesions"
            elif hu==3:
                st="Basal cell carcinoma"
            elif hu==4:
                st="Actinic keratoses"
            elif hu==5:
                st="Vascular lesions"
            elif hu==6:
                st="Dermatofibroma"

            return render_template("index.html",output=st,img_loc=img_location)
if __name__=="__main__":
    app.run(debug=True)