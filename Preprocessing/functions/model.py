from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras.models import Model

def get_image_model_xception():
    xception_model = Xception(weights='imagenet')
    image_model = Model(inputs=[xception_model.input], outputs=[xception_model.layers[-2].output])
    image_model.trainable = False
    return image_model
