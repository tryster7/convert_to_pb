import tensorflow as tf
import tensorflow.keras as keras
import keras
import cv2
import numpy as np
from keras.layers import Input,Dropout
from keras.layers.core import Dense
from keras.models import Model


##################define variables/parameters#################
class_names=["Atelectasis","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis","Effusion","Pneumonia","Pleural_thickening","Cardiomegaly","Nodule Mass","Hernia"]
image_dimension=224

##################define model#################
base_model_class = keras.applications.densenet.DenseNet121

input_shape=(image_dimension, image_dimension, 1)
img_input = Input(shape=input_shape)
base_model = base_model_class(
    include_top=False,
    input_tensor=img_input,
    input_shape=input_shape,
    weights=None,
    )
x = base_model.output
x=keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu',name="dense_512")(x)
x = Dropout(0.5)(x)
predictions = Dense(14, activation="sigmoid", name="my_prediction")(x)
model = Model(inputs=img_input, outputs=predictions)
##################load trained weight in model#################
##################pre-process the image data#################
image_array = cv2.imread('1.png')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_grey = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
cl1 = clahe.apply(img_grey)
cl1 = cl1[..., np.newaxis]
image_array = cl1 / 255.
target_size=224
image_array = cv2.resize(image_array, (224,224))
image_array = image_array[..., np.newaxis]
imagenet_mean = np.array([0.5])
imagenet_std = np.array([0.229])
image_array = (image_array - imagenet_mean) / imagenet_std
##################prediction for the pre-procesed image data#################
image_array=image_array[np.newaxis,...]
y_hat=model.predict(image_array)
final_pred_labels = []
for i in range(0,len(y_hat[0])):
    if(y_hat[0][i]>0.1):
        print(class_names[i])
        final_pred_labels.append(class_names[i])

#export_path='gs://dlaas-model/model/chest/export/1'
#model.load_weights('chest_weights.h5')
#tf.saved_model.save(model, export_dir=export_path)
#with tf.keras.backend.get_session() as sess:
#    tf.saved_model.simple_save(
#        sess,
#        export_path,
#        inputs={'input_image': model.input},
#        outputs={t.name: t for t in model.outputs})
