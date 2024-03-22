import tensorflow as tf

tf.keras.backend.clear_session()

Input =  tf.keras.layers.Input(shape = [50, 200, 1], name = "Input_Img")
Conv1 = tf.keras.layers.Conv2D(kernel_size=(5,5), padding="same", activation='relu', \
                               kernel_initializer="he_normal", filters = 10)(Input)
Conv2 = tf.keras.layers.Conv2D(kernel_size=(3,3), padding="same", activation='relu', \
                               kernel_initializer="he_normal", filters = 10)(Conv1)
Conv3 = tf.keras.layers.Conv2D(kernel_size=(3,3), padding="same", activation='relu', \
                               kernel_initializer="he_normal", filters = 10)(Conv2)
Flatten = tf.keras.layers.Flatten()(Conv3)
BatchNormalization = tf.keras.layers.BatchNormalization()(Flatten)

AlphaDropout1 = tf.keras.layers.AlphaDropout(0.5)(BatchNormalization)
Dense1 = tf.keras.layers.Dense(units = 30, activation = "relu")(AlphaDropout1)
Output1 = tf.keras.layers.Dense(units = 62, activation="softmax",\
                                 kernel_regularizer='l1_l2', bias_regularizer='l1_l2')(Dense1)

AlphaDropout2 = tf.keras.layers.AlphaDropout(0.5)(BatchNormalization)
Dense2 = tf.keras.layers.Dense(units = 30, activation = "relu")(AlphaDropout2)
Output2 = tf.keras.layers.Dense(units = 62, activation="softmax",\
                                 kernel_regularizer='l1_l2', bias_regularizer='l1_l2')(Dense2)

AlphaDropout3 = tf.keras.layers.AlphaDropout(0.5)(BatchNormalization)
Dense3 = tf.keras.layers.Dense(units = 30, activation = "relu")(AlphaDropout3)
Output3 = tf.keras.layers.Dense(units = 62, activation="softmax",\
                                 kernel_regularizer='l1_l2', bias_regularizer='l1_l2')(Dense3)

AlphaDropout4 = tf.keras.layers.AlphaDropout(0.5)(BatchNormalization)
Dense4 = tf.keras.layers.Dense(units = 30, activation = "relu")(AlphaDropout4)
Output4 = tf.keras.layers.Dense(units = 62, activation="softmax",\
                                 kernel_regularizer='l1_l2', bias_regularizer='l1_l2')(Dense4)

AlphaDropout5 = tf.keras.layers.AlphaDropout(0.5)(BatchNormalization)
Dense5 = tf.keras.layers.Dense(units = 30, activation = "relu")(AlphaDropout5)
Output5 = tf.keras.layers.Dense(units = 62, activation="softmax",\
                                 kernel_regularizer='l1_l2', bias_regularizer='l1_l2')(Dense5)

model = tf.keras.Model(inputs = [Input], outputs = [Output1, Output2, Output3, Output4, Output5])

model.compile(optimizer="adam", metrics=["accuracy"],\
              loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy",\
                    "sparse_categorical_crossentropy", "sparse_categorical_crossentropy", \
                    "sparse_categorical_crossentropy"],\
              loss_weights = [0.2, 0.2, 0.2, 0.2, 0.2])