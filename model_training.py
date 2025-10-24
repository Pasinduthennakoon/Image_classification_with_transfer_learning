import tensorflow as tf
import matplotlib.pyplot as plt
import time

#data batch size
BATCH_SIZE = 32
#diamensions of img
IMAGE_SIZE = (128, 128)

#get data to train and test from folder
train_data_dir = "data/cats_dogs/train"
test_data_dir = "data/cats_dogs/test"

#split training data into training and validation
train_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,
                                                         batch_size=BATCH_SIZE,
                                                         image_size=IMAGE_SIZE,
                                                         subset="training",
                                                         validation_split=0.1,
                                                         seed=42)

validation_data = tf.keras.utils.image_dataset_from_directory(train_data_dir,
                                                              batch_size=BATCH_SIZE,
                                                              image_size=IMAGE_SIZE,
                                                              subset="validation",
                                                              validation_split=0.1,
                                                              seed=42)
#get data to test
test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMAGE_SIZE)

#data feature scaling
train_data = train_data.map(lambda x,y:(x/255,y))
validation_data = validation_data.map(lambda x,y:(x/255,y))
test_data = test_data.map(lambda x,y:(x/255,y))

#transfer learning(download pretrained model)
pretrained_model = tf.keras.applications.xception.Xception(include_top=False,
                                                           input_shape=(128, 128, 3),
                                                           weights='imagenet',
                                                           pooling='max')
for layer in pretrained_model.layers:
    layer.trainable = False

#create model
model = tf.keras.models.Sequential()

model.add(pretrained_model)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# model.summary()

#compile model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = ['accuracy'])

#train model
start_time = time.time()
history = model.fit(train_data,
                    epochs=3,
                    validation_data=validation_data)
end_time = time.time()
print(f"Total time for training: {(end_time - start_time):.3f} seconds")

# #plot acuuracy vs validation accuracy
# fig = plt.figure()
# plt.plot(history.history['accuracy'], color='teal', label='accuracy')
# plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=20)
# plt.legend()
# plt.show()
#
# #plot loss vs validation lass
# fig = plt.figure()
# plt.plot(history.history['loss'], color='teal', label='loss')
# plt.plot(history.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend()
# plt.show()

model.save('model.h5')