import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

# Load test dataset
test_data_dir = "data/cats_dogs/test"
test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMAGE_SIZE)

class_names = test_data.class_names

#read image
image = cv2.imread("data/cats_dogs/test/cats/cat.4003.jpg")
# plt.imshow(image)
# plt.show()

#resize image
resize_image = tf.image.resize(image, IMAGE_SIZE)
scaled_image = resize_image/255

# Load the trained model
model = tf.keras.models.load_model("model.h5")

#expand dimention
# np.expand_dims(scaled_image, 0)
y_pred = model.predict(np.expand_dims(scaled_image, 0))
print(y_pred)

if y_pred > 0.5:
    print(class_names[1])
else:
    print(class_names[0])

