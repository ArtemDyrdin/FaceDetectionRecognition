import tensorflow as tf
import numpy as np
import face_recognition

def build_embedding_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), #используем предобученную модель без последнего слоя
                                              include_top=False,
                                              weights='imagenet',
                                              pooling='avg')
    for layer in base.layers:
        layer.trainable = False  # заморозим обучение слоёв, позже разморозим
        
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(base.output) #полносвязный слой с регуляризацией
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1),output_shape = (128,))(x) # нормализуем данные
    #x.shape = (batch_size,128)

    return tf.keras.Model(inputs=base.input, outputs=x)


def load_embedder(model_path):
    model = build_embedding_model()  # создаём модель
    model.load_weights(model_path)
    #model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_face(face_img):
    face_img = tf.image.resize(face_img, (224, 224))
    face_img = tf.cast(face_img, tf.float32) / 255.0
    return np.expand_dims(face_img, axis=0)

def get_face_embedding(model, face_img):
    preprocessed = preprocess_face(face_img)
    embedding = model.predict(preprocessed)[0]
    return embedding
