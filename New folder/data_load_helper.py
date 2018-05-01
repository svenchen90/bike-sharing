import csv
import datetime as dt
import numpy as np

def load_data(x_train, y_train, x_test, y_test):
  classes = ['count']

  # Normalize Data
  mean_image = np.mean(x_train, axis=0)
  x_train -= mean_image
  x_test -= mean_image

  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
    'classes': classes
  }
  return data_dict
	
def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]




# ["instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
def loadData():
	FILE_PATH = "hour.csv"
	keys_attr = ["hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum"]
	keys_label = ["cnt"]
	key_list, attribute_list = loadFile(FILE_PATH)
	
	return getAttr_Label(keys_attr, keys_label, key_list, attribute_list)
	
def loadFile(path):
	DATA_TYPE_MAP = {
		"instant": int,
		"dteday": dateStringToIndex,
		"season": int,
		"yr": int,
		"mnth": int,
		"hr": int,
		"holiday": int,
		"weekday": int,
		"workingday": int,
		"weathersit": int,
		"temp": float,
		"atemp": float,
		"hum": float,
		"windspeed": float,
		"casual": int,
		"registered": int,
		"cnt": int
	}
	key_list = []
	attribute_list = []
	with open(path) as csvfile:
		reader = csv.reader(csvfile) # change contents to floats
		for index,row in enumerate(reader): # each row is a list
			if index == 0:
				key_list = row;
			else:
				attribute_list.append([ DATA_TYPE_MAP[k](row[i]) for i,k in enumerate(key_list)])
	
	return key_list, attribute_list

def getAttr_Label(keys_attr, keys_label, key_list, attribute_list):
	index_list = getIndexList(key_list, keys_attr)
	sub_attr_list = [sublistByIndex(a, index_list) for a in attribute_list]
	
	label_key = getIndexList(key_list, keys_label)
	label_list = [sublistByIndex(a, label_key) for a in attribute_list]

	return np.array(sub_attr_list), np.array(label_list)
	
def dateStringToIndex(str):
	list = str.split('-')
	year  = int(list[0])
	month = int(list[1])
	day = int(list[2])
	return (dt.date(year, month, day) - dt.date(year,1,1)).days + 1

def sublistByIndex(list, list_index):
	try:
		return [ list[i] for i in list_index]
	except:
		print('error - sublistByIndex')
		return []
		
def getIndexList(list, sub_list):
	try:
		return [list.index(k) for k in sub_list]
	except:
		print('error - getIndexList')
		return []