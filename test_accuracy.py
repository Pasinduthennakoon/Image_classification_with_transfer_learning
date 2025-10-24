import tensorflow as tf

# Data batch size
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

# Load test dataset
test_data_dir = "data/cats_dogs/test"

test_data = tf.keras.utils.image_dataset_from_directory(test_data_dir,
                                                        batch_size=BATCH_SIZE,
                                                        image_size=IMAGE_SIZE)

# Normalize (important to match training)
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define metrics
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
accuracy = tf.keras.metrics.BinaryAccuracy()

# Loop through batches
for batch in test_data.as_numpy_iterator():
    x, y = batch
    y_pred = model.predict(x)
    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)
    accuracy.update_state(y, y_pred)

# Display final metrics
print("Precision:", precision.result().numpy())
print("Recall:", recall.result().numpy())
print("Accuracy:", accuracy.result().numpy())

