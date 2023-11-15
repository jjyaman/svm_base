import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Generamos un conjunto de puntos azules aleatorios
x_blue = np.random.rand(100, 2)
x_blue[:, 0] += 1

# Generamos un conjunto de puntos rojos aleatorios
x_red = np.random.rand(100, 2)
x_red[:, 1] += 1

# Combinamos los dos conjuntos de puntos
x = np.vstack((x_blue, x_red))

# Generamos las etiquetas de los puntos
y = np.hstack((np.ones(100), np.zeros(100)))

# Entrenamos la máquina de soporte vectorial, en este caso utilizaremos un kernel lineal
clf = SVC(kernel='linear')
clf.fit(x, y)

# Generamos un nuevo punto
x_new = np.array([0.5, 0.5])
x_new = x_new.reshape(1, 2)

# Clasificamos el nuevo punto
y_pred = clf.predict(x_new)

for i in (y_pred):
    if i >= 1:
        print('Azul')
    else:
        print('Rojo')

# Visualizamos el plano decisional
plt.scatter(x_blue[:, 0], x_blue[:, 1], c='b', label='Azul')
plt.scatter(x_red[:, 0], x_red[:, 1], c='r', label='Rojo')

# Dibujamos el plano decisional
w = clf.coef_[0]
b = clf.intercept_[0]
x_min = min(x[:, 0])
x_max = max(x[:, 0])
y_min = (-b - w[0] * x_min) / w[1]
y_max = (-b - w[0] * x_max) / w[1]
plt.plot([x_min, x_max], [y_min, y_max], c='k')

# Añadimos etiquetas
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
