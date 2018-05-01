import csv
import datetime as dt
import numpy as np

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
		
		

import matplotlib.pyplot as plt
		
FILE_PATH = "hour.csv"
keys_attr = ["hr"]
keys_label = ["cnt"]
key_list, attribute_list = loadFile(FILE_PATH)

hours, ctn = getAttr_Label(keys_attr, keys_label, key_list, attribute_list)

x = [ i for i in range(0,24)]
y = [ 0 for i in range(0,24)]

for index, item in enumerate(hours):
	y[item[0]] += ctn[index][0]

print(y)
# plot
ax = plt.subplot(111)

t = np.array(x)
s = np.array(y)
line, = plt.plot(t, s, lw=2)

# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            # arrowprops=dict(facecolor='black', shrink=0.05),
            # )

#plt.ylim(-2,2)
plt.show()
		