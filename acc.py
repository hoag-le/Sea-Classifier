import sys
sys.path.append('./src')
from data_preparation import load_images
from tensorflow import keras

X_test, y_test, class_names = load_images('data_pp/test')
model = keras.models.load_model('models/sea_classifier.h5')

loss, acc = model.evaluate(X_test, y_test)
print('Test accuracy:', acc)
