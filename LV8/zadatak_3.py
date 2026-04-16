import numpy as np
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt
import cv2
num_classes = 10
input_shape = (28, 28, 1)
x_test = cv2.imread("test_8.png", cv2.IMREAD_GRAYSCALE)
x_test = x_test.astype("float32") / 255
x_test = cv2.resize(x_test, (28, 28))
x_test = np.array([x_test])
y_test = [8]

x_test_s = np.expand_dims(x_test, -1)
print(x_test_s.shape)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

model = load_model("model.keras")
model.summary()

y_test_p = model.predict(x_test_s)

plt.imshow(x_test[0], cmap="gray")
plt.title("Predviđena oznaka: {}; Originalna oznaka: {}"
              .format(np.argmax(y_test_p, axis=1)[0],
                      y_test[0]))
plt.show()