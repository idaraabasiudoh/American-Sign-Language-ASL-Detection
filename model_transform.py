import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('asl_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('asl_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")