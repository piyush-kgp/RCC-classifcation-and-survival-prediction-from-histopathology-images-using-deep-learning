
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Embedding
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.models import Model, Sequential, load_model
# from tensorflow.keras.initializers import Constant
# print(tf.__version__, tf.test.is_gpu_available())


from keras.layers import Input, Dense, GlobalAveragePooling2D, Embedding
from keras.applications import InceptionV3
from keras.models import Model, Sequential, load_model
from keras.initializers import Constant
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks.callbacks import ModelCheckpoint
import numpy as np
from datetime import datetime

from keras_contrib.applications import ResNet

print("GPUs Available: ", K.tensorflow_backend._get_available_gpus())

# Ref: https://github.com/keras-team/keras-contrib/blob/382f6a2b7739064a1281c1cacdb792bb96436f27/keras_contrib/applications/resnet.py#L437
resnet18_net = ResNet(input_shape=(224,224,3), block="basic_block", repetitions=[2, 2, 2, 2], include_top=False)
resnet34_net = ResNet(input_shape=(224,224,3), block="basic_block", repetitions=[3, 4, 6, 3], include_top=False)
resnet50_net = ResNet(input_shape=(224,224,3), block="bottleneck", repetitions=[3, 4, 6, 3], include_top=False)

# resnet18_net = tf.keras.layers.Layer(resnet18_net)
# resnet34_net = tf.keras.layers.Layer(resnet34_net)
# resnet50_net = tf.keras.layers.Layer(resnet50_net)

inception_net = InceptionV3(input_shape=(224,224,3), include_top=False)


def keras_binary_classifier(pretrained_model, input_shape=(224,224,3)):
    inp = Input(input_shape)
    out = pretrained_model(inp)
    out = GlobalAveragePooling2D()(out)
    out = Dense(1, activation="sigmoid")(out)
    return Model(inp, out)

def keras_multiclass_classifier(pretrained_model, num_classes, input_shape=(224,224,3)):
    inp = Input(input_shape)
    out = pretrained_model(inp)
    out = GlobalAveragePooling2D()(out)
    out = Dense(num_classes, activation="softmax")(out)
    return Model(inp, out)

kirc_model = keras_binary_classifier(resnet18_net)
kirp_model = keras_binary_classifier(resnet18_net)
kich_model = keras_binary_classifier(resnet18_net)
subtype_model = keras_multiclass_classifier(resnet18_net, 3)

kirc_model.summary()

sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
kirc_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
kirp_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
kich_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=25,
    validation_split=0.15
)


train_generator = train_datagen.flow_from_directory(
        '/ssd_scratch/cvit/piyush/share1/dataset/Medic_Kidney/KIRC/',
        target_size=(224, 224),
        batch_size=128,
        subset='training',
        class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
        '/ssd_scratch/cvit/piyush/share1/dataset/Medic_Kidney/KIRC/',
        target_size=(224, 224),
        batch_size=128,
        subset='validation',
        class_mode='binary'
)


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0)


history = kirc_model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=[tensorboard_callback, checkpoint, early_stopping],
#         verbose=2
)
