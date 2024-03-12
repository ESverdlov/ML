#Create and complile models
import tensorflow as tf
tf.keras.backend.clear_session()

# Создание модели
def CreateModel():
  tf.random.set_seed(0)
  model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(kernel_size=(5,5), padding="same", activation='relu',
                        kernel_initializer="he_normal", filters = 10, input_shape=[50, 200, 1]),
  tf.keras.layers.Conv2D(kernel_size=(3,3), padding="same", activation='relu',
                         kernel_initializer="he_normal", filters = 10),
  tf.keras.layers.Conv2D(kernel_size=(3,3), padding="same", activation='relu',
                         kernel_initializer="he_normal", filters = 10),
  tf.keras.layers.Flatten(),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.AlphaDropout(0.5),
  tf.keras.layers.Dense(units = 30, activation = "relu"),
  tf.keras.layers.Dense(units=62, activation="softmax", kernel_regularizer='l1_l2', bias_regularizer='l1_l2')
  ])
  model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  return model

#Первый символ(модель)
model1 = CreateModel()
model1.summary()

#Второй символ(модель)
model2 = CreateModel()
model2.summary()

#Третий символ(модель)
model3 = CreateModel()
model3.summary()

#Четвертый символ(модель)
model4 = CreateModel()
model4.summary()

#Пятый символ(модель)
model5 = CreateModel()
model5.summary()