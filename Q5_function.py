import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

category = ['cat', 'dog']

def show_tensorboard():
    show_img=image.load_img('tensorboard10.PNG')
    show_img.show()

def show_acc():
    img1 = image.load_img('train_acc.PNG')
    img1.show()
    img2 = image.load_img('val_acc.PNG')
    img2.show()

def test_image(n):
    n = n if n > 0 else 1
    n = n + 11249
    c_or_d = random.randint(0,1)
    model = load_model('model_1217.h5')
    img_path = f"dataset/test/{'cat' if c_or_d==0 else 'dog'}/{n}.jpg"
    img_test = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img_test)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    pred_class = 'cat' if pred[0] > pred[1] else 'dog'
    title = 'Class: '+category[c_or_d]+'\n Predict class :'+pred_class
    origin_img = image.load_img(img_path)
    plt.title(title)
    plt.imshow(origin_img)
    plt.show()
    #print(pred)
    #img_test.show()

#test_image(250)
#show_acc()
