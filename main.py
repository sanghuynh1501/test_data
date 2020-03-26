import csv
import numpy as np
import tensorflow as tf

def parse_float(x):
    x = x.replace(",", ".")
    return float(x)

x_test = []
y_test = []

def data_agument(x, y):
    start = 1
    end = 3
    random_number = 10
    x_result = []
    y_result = []
    for i in range(len(x)):
        x_test.append(x[i])
        y_test.append(y[i])
        x_result.append(x[i])
        y_result.append(y[i])

        # if y[i][1] < 3.1:
        #     percents = np.arange(float(start), float(end), float(end - start) / random_number).tolist()
        #     for percent in percents:
        #         x_result.append(x[i])
        #         y_result.append(y[i] - y[i] * percent / 100)
        #         x_result.append(x[i])
        #         y_result.append(y[i] + y[i] * percent / 100)
    return x_result, y_result

x_train = []
y_train = []
with open('test_data_230320.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x = np.array([parse_float(row[1]), parse_float(row[5])])
        y = np.array([parse_float(row[2]), parse_float(row[6])])
        x_train.append(x)
        y_train.append(y)

x_train, y_train = data_agument(x_train, y_train)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)

x_train = x_train[indices]
y_train = y_train[indices]

model = tf.keras.models.Sequential([
  tf.keras.layers.Input((2,)),
  tf.keras.layers.Dense(units=200, use_bias=False),
  tf.keras.layers.Dense(units=2)
])

checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer='adam', metrics=['accuracy'])

# model.load_weights("model87.h5")

model.fit(x_train, y_train / 3, epochs=300, callbacks=[checkpoint], batch_size=5)

# model.load_weights("model.h5")

count = 0
for i in range(len(x_test)):
    x = x_test[i]
    x = np.expand_dims(x, 0)
    y = model.predict(x) * 3
    print(x, y, y_test[i], (abs(y - y_test[i]) / y_test[i]) * 100)

    if ((abs(y[0, 0] - y_test[i, 0]) / y_test[i, 0]) * 100) < 5 and ((abs(y[0, 1] - y_test[i, 1]) / y_test[i, 1]) * 100) < 5:
      count += 1
    
print(count / len(x_test) * 100)

for layer in model.layers:
    weights = layer.get_weights()
    print("weights ", weights)