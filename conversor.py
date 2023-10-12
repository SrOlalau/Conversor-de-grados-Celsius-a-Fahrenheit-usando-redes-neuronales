#Importamos los modulos a usar
#Con respecto a tensorflow lo tengo configurado par que haga uso de la GPU
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Definimos los datos de entrenamiento, en este casos son 2 listas usando el modulo de numpy
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

#Definimos la variable modelo para dar los atributos que queremos que nuestro modelo tenga
#Uso capas densas con el objetivo de que todas las neuronas de las diferentes capas se comuniquen entre si
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, input_shape=[1]),  # Capa de entrada con 1 neurona 
    tf.keras.layers.Dense(units=3, activation='relu'),  # Capa oculta con 1 neuronas y función de activación ReLU
    tf.keras.layers.Dense(units=1)  # Capa de salida con 1 neurona
])

#Utilizamos el optimizador Adam y ajustamos la tasa de aprendizaje a un valor lo suficientemente pequeño como para que pueda correguir con la precision necesaria
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

#Un simple print para saber cuando comienza y acaba el entrenamiento
#Hacemos que haga el entrenamiento durante 1000 generaciones, dependiendo de las capas, neuronas y optimizador varia en cual generacion deja practicamente de aprender
print("Comenzamos el entrenamiento")
historial = modelo.fit(celsius,fahrenheit,epochs=1000,verbose=False)
print("Modelo entrenado")

#Simple gráfico con el objetivo de ver hasta cual generacion ha logrado progreso y aprendido
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

#Comprobamos las prediciones del modelo poniendo un ejemplo y haciendo que lo imprima
print("Hagamos una predicion")
resultado = modelo.predict([100.0])
print(resultado)

#Guardamos el modelo en este formato para posteriormente poder convertirlo a un formato que tensorflowjs pueda usar y asi emplearlo en mi página web
modelo.save("Conversor_grados.h5")

#Exportamos el modelo al formato necesario
!tensorflowjs_converter --input_format keras Conversor_grados.h5 export
