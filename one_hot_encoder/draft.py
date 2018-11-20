#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv

from os import path, makedirs
from sklearn.model_selection import train_test_split


image_name_and_feature_list = []
with open('../raw_data/train.csv') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    for row in raw_data:
        if row[0] == 'Id':
        	continue
        row[1] = [ int(feature_num_str) for feature_num_str in row[1].split()]
        image_name_and_feature_list.append(row)

image_name_to_one_hot = {}
for image_name, feature_list in image_name_and_feature_list:
	feature_one_hot = [ 0 for _ in range(28)]
	for feature_idx in feature_list:
		feature_one_hot[feature_idx] = 1
	image_name_to_one_hot[image_name] = feature_one_hot

image_idx_to_image_name = {}
image_idx_list = []
for image_idx, (image_name, _) in enumerate(image_name_and_feature_list):
	image_idx_to_image_name[image_idx] = image_name
	image_idx_list.append(image_idx)


train, test_and_validation = train_test_split(image_idx_list, test_size=0.40, random_state=42)
test, validation = train_test_split(test_and_validation, test_size=0.50, random_state=42)

np.savetxt('test_idx.txt', np.array(test), fmt='%d')
image_names = []
for image_idx in test:
	image_names.append(image_idx_to_image_name[image_idx])
np.savetxt('test_lables.txt', np.array(image_names), fmt='%s')

one_hot = []
for image_idx in test:
	one_hot.append(image_name_to_one_hot[image_idx_to_image_name[image_idx]])
np.save('test_one_hot.npy', np.array(one_hot))

d = np.load('test_one_hot.npy')

print(image_name_and_feature_list[test[0]])
print(d == np.array(one_hot))
print(d[0])
# print(image_name_and_feature_list[0])
# print(image_idx_list[0])
# print(image_idx_to_image_name[0])
# print(image_name_to_one_hot[image_idx_to_image_name[0]])