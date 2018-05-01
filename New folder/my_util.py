import numpy as np


def load_data(x_train, y_train, x_test, y_test):
  classes = [0,1,2,3,4,5,6,7,8,9]

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

