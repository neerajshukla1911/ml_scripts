import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import json
from os import path
from keras.applications.inception_v3 import preprocess_input
import csv

prediction_csv_filepath = 'sleeve_length_women_prediction.csv'

test_image_base_dir = '/home/neeraj/data/sleeve_length/women/test'
model_def_filepath = '/home/neeraj/keras_jobs/sleeve_length_women/inception_v3.def'
model_weights_filepath = '/home/neeraj/keras_jobs/sleeve_length_women/weights.15-0.39.hdf5'
model_cls_filepath = '/home/neeraj/keras_jobs/sleeve_length_women/inception_v3.cls'
class_indices = json.load(open(model_cls_filepath, 'r'))
exchanged_class_indices = {v:k for k,v in class_indices.items()}

json_file = open(model_def_filepath, "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)


model.load_weights(model_weights_filepath)

print("loaded model from disk")

grayscale = False
if model.input_shape[3] == 1:
    grayscale = True

# #for single image prediction use below code
# im = load_img('/home/neeraj/data/sleeve_length/women/test/Half/64626064cab1871d9257e4650a436301af3f9894d165b.jpg', grayscale=grayscale, target_size=(model.input_shape[1], model.input_shape[2]))
# im_arr = img_to_array(im)
# im_arr = np.expand_dims(im_arr, axis=0)
# im_arr = preprocess_input(im_arr)
# predictions = model.predict(im_arr)

# index_max = np.argmax(predictions)
# prediction_labels  = {}
# for idx, val in enumerate(list(predictions[0])):
#     prediction_labels[exchanged_class_indices[idx]] = float(val)
# print(prediction_labels)
# print(exchanged_class_indices[index_max])

predictions_list = []
for dir_name in os.listdir(test_image_base_dir):
    print("predicting images from dir {}".format(dir_name))
    count = 0
    dir_path = '{}/{}'.format(test_image_base_dir,dir_name)
    for f in os.listdir(dir_path):
        filpath = '{}/{}'.format(dir_path, f)
        if path.isfile(filpath):
            im = load_img(filpath, grayscale=grayscale, target_size=(model.input_shape[1], model.input_shape[2]))
            im_arr = img_to_array(im)
            im_arr = np.expand_dims(im_arr, axis=0)
            im_arr = preprocess_input(im_arr)
            predictions = model.predict(im_arr)
            index_max = np.argmax(predictions)
            predicted = exchanged_class_indices[index_max]
            predictions_list.append([dir_name, predicted])
            count += 1
            if count%50 == 0:
                print("predicted {} images so far".format(count))
print("Done all predictions")
print(predictions_list)
print("Saving predictions to csv file")

csv_file = open(prediction_csv_filepath, 'w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['actual', 'predicted'])
csv_writer.writerows(predictions_list)
csv_file.close()