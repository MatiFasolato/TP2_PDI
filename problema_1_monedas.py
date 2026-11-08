import cv2
import numpy as np
import matplotlib.pyplot as plt


# Cargar la imagen
img = cv2.imread("monedas.jpg")

# Verificar que se haya cargado correctamente
if img is None:
    print("Error: no se pudo cargar la imagen.")
else:
    print("Imagen cargada correctamente.")
    print("Dimensiones:", img.shape)
    

# Convertir de BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar imagen original
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")
plt.show()


# Convertir a escala de grises
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mostrar resultado
plt.figure(figsize=(10, 6))
plt.imshow(img_gris, cmap="gray")
plt.title("Imagen en escala de grises")
plt.axis("off")
plt.show()


# Niveles de intensidad de la imagen en escala de grises
plt.figure(figsize=(10, 5))
plt.hist(img_gris.ravel(), bins=256, range=(0, 256))

plt.title("Histograma de la imagen en escala de grises")
plt.xlabel("Nivel de intensidad")
plt.ylabel("Cantidad de píxeles")
plt.show()