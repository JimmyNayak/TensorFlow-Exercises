# import tensorflow as tf
#
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
#
# a = tf.constant(10)
# b = tf.constant(32)
# print(sess.run(a + b))
#
# sess.close()


# import tensorflow as tf
# import pandas as pd
#
# feature_columns = [[140, 1], [130, 1], [150, 0], [170, 0]]
# feature_labels = ["Apple", "Apple", "orange", "orange"]
#
# classifier = tf.estimator.LinearClassifier(feature_columns)
#
# classifier.train(input_fn=170)
#
# prediction = classifier.predict(input_fn=predict_input_fn)


import pandas as pd
import numpy as np

print(pd.__version__)

# Example of series (one column data)
# pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
#################################################################

# # Example of DataFrame
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
pd.DataFrame({'City name': city_names, 'Population': population})

##################################################################

# But most of the time, you load an entire file into a DataFrame. The following example loads a file with California
# housing data. Run the following cell to load the data and create feature definitions:


california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
                                           sep=",")
california_housing_dataframe.describe()

california_housing_dataframe.head()
print(california_housing_dataframe.head())

california_housing_dataframe.hist('housing_median_age')

##################################################################
# You can access DataFrame data using familiar Python dict/list operations:
cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(type(cities['City name']))
var = cities['City name']

print(type(cities['City name'][1]))
var1 = cities['City name'][1]

print(type(cities[0:2]))
var2 = cities[0:2]
##################################################################

# Manipulating Data
np.log(population / 1000)

##################################################################
# Exercise #1

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
saint_name = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
area = pd.Series([53, 52, 51])
cities = pd.DataFrame(
    {'City name': city_names, 'Population': population, 'Saint Name': saint_name, 'Area square miles': area})

cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(
    lambda name: name.startswith('San'))
print(cities['Is wide and has saint name'])
print(cities.reindex([2, 0, 1]))
