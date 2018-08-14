from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score
import json

predicted_csv_filepath = '/home/neeraj/keras_jobs/sleeve_length_men_final_weights_prediction.csv'
class_indices_filepath = "/home/neeraj/keras_jobs/sleeve_length_men4/inception_v3.cls"

class_indices = json.load(open(class_indices_filepath, 'r'))
classes = list(class_indices.keys())

df = pd.read_csv(predicted_csv_filepath)
print("Accuracy score: {}".format(accuracy_score(df.actual, df.predicted)))
print(classes)
conf_arr = confusion_matrix(df.actual,df.predicted, labels=classes)
sn.set(font_scale=1)#for label size
sn.heatmap(conf_arr, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)# font size
plt.show()
# plt.savefig('hi.png')