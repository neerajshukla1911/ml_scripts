# coding: utf-8

# In[10]:

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#from keras import backend as K
from keras.layers.core import K
K.set_learning_phase(0)

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter

from keras.models import model_from_json
from keras.models import load_model
import shutil
model_def_filepath = '/home/neeraj/keras_jobs/pattern_new_6_cat_balanced/inception_v3.def'
model_weights_filepath = '/home/neeraj/keras_jobs/pattern_new_6_cat_balanced/final_pattern_new_6_cat_balanced_inception_v3.h5'

json_file = open(model_def_filepath, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_filepath)

#loaded_model =  load_model('/home/neeraj/keras_jobs/pattern_biased_polka_new/final_pattern_biased_polka_new_inception_v3.h5')
print("loaded model from disk")

if os.path.isdir("./export"):
    shutil.rmtree("./export")

model = loaded_model

# export model

export_path = "export/pattern/1"

builder = saved_model_builder.SavedModelBuilder(export_path)

print(model.input)
print(model.output)

signature = predict_signature_def(inputs={'images': model.input},
                                  outputs={'scores': model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()