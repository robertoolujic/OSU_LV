import numpy as np
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt

num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

model = load_model("model.keras")
model.summary()

y_test_p = model.predict(x_test_s)

bad_predictions = np.where(np.argmax(y_test_p, axis=1) != y_test)[0]
print("Broj krivo klasificiranih slika:", len(bad_predictions))
for i in range(0,9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[bad_predictions[i]], cmap="gray")
    plt.title("Predviđena oznaka: {}\n; Originalna oznaka: {}"
              .format(np.argmax(y_test_p[bad_predictions[i]]),
                      y_test[bad_predictions[i]]))
plt.tight_layout()
plt.show()