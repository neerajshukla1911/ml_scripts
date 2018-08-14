import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.engine.training import Model
import tensorflow as tf
from PIL import Image
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math
from keras.regularizers import l2
import numpy as np
import json

K.clear_session()
# base dir where all jobs will be saved
jobs_base_dir = '/path/to/keras_jobs'

if not os.path.exists(jobs_base_dir):
    os.makedirs(jobs_base_dir)

# name of job
job_name = 'demo_job'
# training images dir
train_dir = "/path/to/train/image/dir"
# validation images dir
validation_dir = "/path/to/validation/image/dir"
batch_size = 40
img_width, img_height = 128, 128
input_tensor = Input(shape=(img_width, img_height, 3))
epochs = 50

model_name = 'inception_v3'
job_path = "{}/{}".format(jobs_base_dir, job_name)
tensorboard_dir = "{}/{}".format(job_path, "tensorboard")

if not os.path.exists(jobs_base_dir):
    os.makedirs(jobs_base_dir)

if not os.path.exists(job_path):
    os.makedirs(job_path)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

train_class_list = os.listdir(train_dir)
validation_class_list = os.listdir(validation_dir)

if len(train_class_list) != len(validation_class_list):
    raise Exception("Train image class list should be equal to validataion image class list")

nb_classes = len(train_class_list)
nb_train_samples = 0
nb_validation_samples = 0

for val in train_class_list:
    nb_train_samples += len(next(os.walk("{}/{}".format(train_dir, val)))[2])
for val in validation_class_list:
    nb_validation_samples += len(next(os.walk("{}/{}".format(validation_dir, val)))[2])

print("Total training images: {}".format(nb_train_samples))
print("Total validation images: {}".format(nb_validation_samples))


def lr_schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


weights_path = "{}/{}".format(job_path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
csv_logger = CSVLogger("{}/{}.log".format(job_path, model_name))
tensorboard = TensorBoard(log_dir="{}".format(tensorboard_dir), histogram_freq=2, batch_size=32, write_graph=True,
                          write_grads=False,
                          write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
lr_scheduler = LearningRateScheduler(lr_schedule)


def preprocess_function(img):
    # whites out bottom half and face
    arr = np.copy(img)
    x = Image.fromarray(arr.astype('uint8'), 'RGB')
    img_w, img_h = x.size
    background = Image.new('RGBA', (img_w, img_h // 2), (255, 255, 255, 255))
    background2 = Image.new('RGBA', (img_w, img_h // 6), (255, 255, 255, 255))
    x.paste(background, (0, img_h // 2))
    x.paste(background2, (0, 0))
    x = np.asarray(x)
    x.setflags(write=True)
    x = np.float_(x)
    x *= 1. / 255
    # ends whites out bottom half and face
    return x


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #         width_shift_range=.35,
    #         channel_shift_range= 34,
    rotation_range=15,
    horizontal_flip=0.2,
    vertical_flip=0.2,
    shear_range=0.2)
#         preprocessing_function=preprocess_function)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size, )

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size, )

#     pl = tf.placeholder(tf.float32, shape=(img_height, img_width, 3))
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = GlobalAveragePooling2D()(base_model.output)
# x = Dropout(.4)(x)
#     x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(base_model.input, predictions)

#     model = load_model(filepath='./model4.29-0.69.hdf5')

opt = SGD(lr=.01, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model_json = model.to_json()
with open("{}/{}.def".format(job_path, model_name), "w") as json_file:
    json_file.write(model_json)

print(train_generator.class_indices)
f = open("{}/{}.cls".format(job_path, model_name), 'w')
f.write(json.dumps(train_generator.class_indices))
f.close()

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples, callbacks=[lr_scheduler, csv_logger, checkpointer, tensorboard])

model.save_weights("{}/final_{}_{}.h5".format(job_path, job_name, model_name))