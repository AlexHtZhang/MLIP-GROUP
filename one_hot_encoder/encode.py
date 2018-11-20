#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" @class: Encode
	generate training, testing, validation data lables set as .npy files. Initalize with train.csv from kaggle.
"""

__author__ = "AlexHtZhang"
__copyright__ = "Copyright 2018, MLIP Project"

__license__ = "MIT"
__version__ = "1.0"
__status__ = "Beta"

import numpy as np
import csv

from os import path, makedirs
from sklearn.model_selection import train_test_split

"""	@class: Encode
	generate training, testing, validation data lables set as .txt files and corresponding one hot keys as .npy files.

	func __init__(self, train_csv_path, output_folder_path, train_rate, validation_rate, test_rate):
	@param train_csv_path: something like '../raw_data/train.csv'
	@type: str
	@param output_folder_path: something like '../processed_data/'
	@type: str
	@param train_rate: from 0 to 1. input test + validation + train should be equal to 1.
	@type: float
	@param validation_rate: from 0 to 1. input test + validation + train should be equal to 1.
	@type: float
	@param test_rate: from 0 to 1. input test + validation + train should be equal to 1.
	@type: float
"""
class Encode:

	def __init__(self, train_csv_path, output_folder_path, train_rate, validation_rate, test_rate):

		assert isinstance(train_csv_path, str) # input train_csv_path should be python buildin str
		assert isinstance(output_folder_path, str) # input output_folder_path should be python buildin str
		assert isinstance(train_rate, float) # input train_rate should be python buildin float
		assert isinstance(validation_rate, float) # input validation_rate should be python buildin float
		assert isinstance(test_rate, float) # input test_rate should be python buildin float
		assert train_rate + validation_rate + test_rate == 1 # input test + validation + train should be equal to 1.

		image_name_and_feature_list = self.__get_image_name_and_feature_list(train_csv_path) # return list
		image_name_to_one_hot = self.__get_image_name_to_one_hot(image_name_and_feature_list) # return dictionary
		image_idx_list, image_idx_to_image_name = self.__get_image_idx_list_and_image_idx_to_image_name(image_name_and_feature_list) # return (list, dictonary)
		train, validation, test = self.__split_train_val_test(image_idx_list, validation_rate, test_rate) # return triplet lists of idx(int)

		self.__check_output_folder(output_folder_path) # output folder non-exist create one.
		self.__save_names_in_txt(output_folder_path, train, validation, test, image_idx_to_image_name)
		self.__save_one_hot_encoding(output_folder_path, train, validation, test, image_name_to_one_hot, image_idx_to_image_name)

	def __get_image_name_and_feature_list(self, train_csv_path):

		image_name_and_feature_list = []

		with open(train_csv_path) as raw_data_file:

		    raw_data = csv.reader(raw_data_file, delimiter=',')

		    for row in raw_data:
		        if row[0] == 'Id':
		        	continue
		        row[1] = [ int(feature_num_str) for feature_num_str in row[1].split()]
		        image_name_and_feature_list.append(row)

		return image_name_and_feature_list

	def __get_image_name_to_one_hot(self, image_name_and_feature_list):

		image_name_to_one_hot = {}
		for image_name, feature_list in image_name_and_feature_list:
			feature_one_hot = [ 0 for _ in range(28)]
			for feature_idx in feature_list:
				feature_one_hot[feature_idx] = 1
			image_name_to_one_hot[image_name] = feature_one_hot

		return image_name_to_one_hot

	def __get_image_idx_list_and_image_idx_to_image_name(self, image_name_and_feature_list):

		image_idx_to_image_name = {}
		image_idx_list = []
		for image_idx, (image_name, _) in enumerate(image_name_and_feature_list):
			image_idx_to_image_name[image_idx] = image_name
			image_idx_list.append(image_idx)

		return image_idx_list, image_idx_to_image_name

	def __split_train_val_test(self, image_idx_list, validation_rate, test_rate):

		train, test_and_validation = train_test_split(image_idx_list, test_size=validation_rate+test_rate, random_state=42)
		test, validation = train_test_split(test_and_validation, test_size=validation_rate/(validation_rate+test_rate), random_state=42)

		return train, validation, test

	def __check_output_folder(self, output_folder_path):

		if not path.exists(output_folder_path):
			makedirs(output_folder_path)

		return

	def __save_names_in_txt(self, output_folder_path, train, validation, test, image_idx_to_image_name):

		# lazy coding style
		np.savetxt(output_folder_path + 'train_idx.txt', np.array(train), fmt='%d')
		image_names = []
		for image_idx in test:
			image_names.append(image_idx_to_image_name[image_idx])
		np.savetxt(output_folder_path + 'train_lables.txt', np.array(image_names), fmt='%s')

		np.savetxt(output_folder_path + 'validation_idx.txt', np.array(validation), fmt='%d')
		image_names = []
		for image_idx in test:
			image_names.append(image_idx_to_image_name[image_idx])
		np.savetxt(output_folder_path + 'validation_lables.txt', np.array(image_names), fmt='%s')

		np.savetxt(output_folder_path + 'test_idx.txt', np.array(test), fmt='%d')
		image_names = []
		for image_idx in test:
			image_names.append(image_idx_to_image_name[image_idx])
		np.savetxt(output_folder_path + 'test_lables.txt', np.array(image_names), fmt='%s')

		return

	def __save_one_hot_encoding(self, output_folder_path, train, validation, test, image_name_to_one_hot, image_idx_to_image_name):

		# lazy coding style
		one_hot = []
		for image_idx in train:
			one_hot.append(image_name_to_one_hot[image_idx_to_image_name[image_idx]])
		np.save(output_folder_path + 'train_one_hot.npy', np.array(one_hot))

		one_hot = []
		for image_idx in validation:
			one_hot.append(image_name_to_one_hot[image_idx_to_image_name[image_idx]])
		np.save(output_folder_path + 'validation_one_hot.npy', np.array(one_hot))

		one_hot = []
		for image_idx in test:
			one_hot.append(image_name_to_one_hot[image_idx_to_image_name[image_idx]])
		np.save(output_folder_path + 'test_one_hot.npy', np.array(one_hot))

		return


# test
if __name__ == "__main__":

	Encode('../raw_data/train.csv', '../processed_data/', 0.6, 0.2, 0.2)
