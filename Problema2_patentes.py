import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Carpeta de resultados ----------------------------------------

CARPETA_RESULTADOS = "Resultados"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


def guardar_figura(nombre):
    plt.savefig(
        os.path.join(CARPETA_RESULTADOS, nombre + ".png"),
        dpi=150,
        bbox_inches="tight"
    )


# --- Cargamos la imagen ----------------------------------------------

img = cv2.imread("Patentes/img01.png")              # OpenCV la lee en BGR

print("Dimensiones:", img.shape)                     # (alto, ancho, canales)
print("Tipo de dato:", img.dtype)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       # para mostrarla bien con matplotlib
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # para procesarla

print("Valor mínimo y máximo en grises:", img_gris.min(), img_gris.max())

# --- Visualización ------------------------------------------------

plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_gris, cmap="gray", vmin=0, vmax=255)
plt.title("Escala de grises")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_original_gris")
plt.show(block=True)

# --- Histograma ---------------------------------------------------

plt.figure()
plt.hist(img_gris.ravel(), bins=256, range=(0, 256))
plt.title("Histograma de la imagen en escala de grises")
plt.xlabel("Nivel de intensidad")
plt.ylabel("Cantidad de píxeles")
guardar_figura("p2_histograma")
plt.show(block=True)

# --- Umbralado ----------------------------------------------------

T_otsu, img_bin = cv2.threshold(
    img_gris,
    thresh=0,
    maxval=255,
    type=cv2.THRESH_OTSU
)

print("Umbral calculado por Otsu:", T_otsu)
print("Valores después del umbralado:", np.unique(img_bin))

# --- Visualización del umbralado ----------------------------------

plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
plt.imshow(img_gris, cmap="gray", vmin=0, vmax=255)
plt.title("Escala de grises")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_bin, cmap="gray", vmin=0, vmax=255)
plt.title(f"Umbralado - Otsu T={T_otsu:.0f}")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_umbralado_otsu")
plt.show(block=True)

# --- Componentes conectadas ---------------------------------------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    img_bin,
    connectivity=4,
    ltype=cv2.CV_32S
)

print("Cantidad de objetos encontrados:", num_labels - 1)   # la etiqueta 0 es el fondo

# --- Visualización de las componentes -----------------------------

img_componentes = img_rgb.copy()

for i in range(1, num_labels):                  
    x, y, w, h, area = stats[i]
    cv2.rectangle(img_componentes, (x, y), (x + w, y + h), (255, 0, 0), 1)

plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
plt.imshow(labels, cmap="jet")
plt.title(f"Etiquetas ({num_labels - 1} objetos)")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_componentes)
plt.title("Rectángulo de cada objeto")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_componentes")
plt.show(block=True)


# --- Filtrado por forma de letra ----------------------------------

candidatos = []

for i in range(1, num_labels):          
    x, y, w, h, area = stats[i]

    alto_ok = 8 <= h <= 25                  # tamaño de una letra
    aspecto_ok = 1.2 <= h / w <= 3.5        # más alta que ancha

    if alto_ok and aspecto_ok:
        candidatos.append((x, y, w, h, area))

print("Objetos con forma de letra:", len(candidatos))



# --- Visualización de los candidatos ------------------------------

img_candidatos = img_rgb.copy()

for x, y, w, h, area in candidatos:
    cv2.rectangle(img_candidatos, (x, y), (x + w, y + h), (255, 0, 0), 1)

plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
plt.imshow(img_componentes)
plt.title(f"Todos los objetos ({num_labels - 1})")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_candidatos)
plt.title(f"Con forma de letra ({len(candidatos)})")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_candidatos")
plt.show(block=True)


# --- Agrupamiento: letras alineadas -------------------------------

mejor_grupo = []

for x1, y1, w1, h1, a1 in candidatos:          # candidato de referencia
    grupo = []

    for x2, y2, w2, h2, a2 in candidatos:      # lo comparamos con todos
        misma_altura = abs(h2 - h1) <= 4
        misma_fila = abs((y2 + h2 / 2) - (y1 + h1 / 2)) <= h1
        cerca = abs(x2 - x1) <= 90            # una patente mide menos de 100 px de ancho

        if misma_altura and misma_fila and cerca:
            grupo.append((x2, y2, w2, h2, a2))

    if abs(len(grupo) - 6) < abs(len(mejor_grupo) - 6):
        mejor_grupo = grupo

mejor_grupo = sorted(mejor_grupo)                # ordenamos de izquierda a derecha

print("Letras en el grupo:", len(mejor_grupo))

# --- Visualización del grupo --------------------------------------

img_grupo = img_rgb.copy()

for x, y, w, h, area in mejor_grupo:
    cv2.rectangle(img_grupo, (x, y), (x + w, y + h), (255, 0, 0), 1)

plt.figure(figsize=(12, 4))

ax1 = plt.subplot(121)
plt.imshow(img_candidatos)
plt.title(f"Candidatos ({len(candidatos)})")
plt.xticks([])
plt.yticks([])

plt.subplot(122, sharex=ax1, sharey=ax1)
plt.imshow(img_grupo)
plt.title(f"Letras alineadas ({len(mejor_grupo)})")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_letras_alineadas")
plt.show(block=True)



# --- Pruebas de umbral --------------------------------------------

# Con Otsu (T=114) encontramos solo 4 letras: algunas quedan pegadas
# al marco blanco de la placa. Como el marco es un poco más oscuro que las
# letras, probamos mover el umbral.
# Para eso armamos funciones con los anteriores y así repetirlos.


def buscar_letras(img_bin):
    """componentes conectadas + filtro por forma de letra."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_bin,
        connectivity=4,
        ltype=cv2.CV_32S
    )

    candidatos = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        alto_ok = 8 <= h <= 25
        aspecto_ok = 1.2 <= h / w <= 3.5
        if alto_ok and aspecto_ok:
            candidatos.append((x, y, w, h, area))

    return candidatos


def agrupar_letras(candidatos):
    """grupo de letras alineadas más cercano a 6."""
    mejor_grupo = []

    for x1, y1, w1, h1, a1 in candidatos:
        grupo = []
        for x2, y2, w2, h2, a2 in candidatos:
            misma_altura = abs(h2 - h1) <= 4
            misma_fila = abs((y2 + h2 / 2) - (y1 + h1 / 2)) <= h1
            cerca = abs(x2 - x1) <= 90
            if misma_altura and misma_fila and cerca:
                grupo.append((x2, y2, w2, h2, a2))

        if abs(len(grupo) - 6) < abs(len(mejor_grupo) - 6):
            mejor_grupo = grupo

    return sorted(mejor_grupo)


# --- Probamos umbrales por debajo y por encima de Otsu --------------

umbrales = [85, 100, T_otsu, 130, 145, 160]          
print("Umbral de Otsu:", T_otsu)

# Zona de la patente para el zoom (alrededor de las letras que encontro otsu)
alto = max(h for x, y, w, h, area in mejor_grupo)
zx0 = max(min(x for x, y, w, h, area in mejor_grupo) - 3 * alto, 0)
zy0 = max(min(y for x, y, w, h, area in mejor_grupo) - 2 * alto, 0)
zx1 = max(x + w for x, y, w, h, area in mejor_grupo) + 3 * alto
zy1 = max(y + h for x, y, w, h, area in mejor_grupo) + 2 * alto

T_elegido = None
plt.figure(figsize=(12, 6))

for k, T in enumerate(umbrales):
    _, img_bin_T = cv2.threshold(img_gris, T, 255, cv2.THRESH_BINARY)
    grupo = agrupar_letras(buscar_letras(img_bin_T))
    print(f"Umbral {T:.0f}: {len(grupo)} letras")

    if len(grupo) == 6 and T_elegido is None:   # nos quedamos con el primero que da 6
        T_elegido = T

    zona = cv2.cvtColor(img_bin_T[zy0:zy1, zx0:zx1], cv2.COLOR_GRAY2RGB)
    for x, y, w, h, area in grupo:
        cv2.rectangle(zona, (x - zx0, y - zy0), (x - zx0 + w, y - zy0 + h), (255, 0, 0), 1)

    plt.subplot(2, 3, k + 1)
    plt.imshow(zona)
    plt.title(f"T={T:.0f}: {len(grupo)} letras")
    plt.xticks([])
    plt.yticks([])

plt.suptitle(f"img01: letras encontradas según el umbral (Otsu = {T_otsu:.0f})")
guardar_figura("p2_umbrales_probados")
plt.show(block=True)

# Bajar el umbral empeora (la placa entera queda blanca y todo se une).
# Subirlo separa las letras del marco: el primer umbral con 6 letras es
# el que usamos como respaldo cuando Otsu falla.

_, img_bin_final = cv2.threshold(img_gris, T_elegido, 255, cv2.THRESH_BINARY)
grupo_final = agrupar_letras(buscar_letras(img_bin_final))

print("Umbral de respaldo elegido:", T_elegido, "->", len(grupo_final), "letras")




# --- Rectángulo de la patente (Punto A) ---------------------------

# Rectángulo que contiene a las 6 letras
x_min = min(x for x, y, w, h, area in grupo_final)
y_min = min(y for x, y, w, h, area in grupo_final)
x_max = max(x + w for x, y, w, h, area in grupo_final)
y_max = max(y + h for x, y, w, h, area in grupo_final)

# Lo agrandamos una altura de letra para cada lado, para abarcar toda la placa
alto_letra = max(h for x, y, w, h, area in grupo_final)

x0 = max(x_min - alto_letra, 0)
x1 = x_max + alto_letra
y0 = max(y_min - alto_letra, 0)
y1 = y_max + alto_letra

print("Patente en: x =", x0, "a", x1, " y =", y0, "a", y1)

# Recorte de la patente (ROI)
patente_rgb = img_rgb[y0:y1, x0:x1]
patente_gris = img_gris[y0:y1, x0:x1]

# --- Visualización ------------------------------------------------

img_patente = img_rgb.copy()
cv2.rectangle(img_patente, (x0, y0), (x1, y1), (0, 255, 0), 2)

plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.imshow(img_patente)
plt.title("Patente detectada")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(patente_rgb)
plt.title("Patente recortada")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_patente")
plt.show(block=True)




# --- Segmentación de caracteres (Punto B) -------------------------

#el mismo criterio que usamos para encontrar la patente en la imagen completa nos sirve para separar los caracteres dentro del recorte

# Umbralamos la patente recortada con el mismo umbral
_, patente_bin = cv2.threshold(patente_gris, T_elegido, 255, cv2.THRESH_BINARY)

# Buscamos los caracteres con las mismas funciones de antes
caracteres = agrupar_letras(buscar_letras(patente_bin))

print("Caracteres segmentados:", len(caracteres))


# --- Visualización de la patente segmentada -----------------------

patente_caracteres = patente_rgb.copy()

for x, y, w, h, area in caracteres:
    cv2.rectangle(patente_caracteres, (x, y), (x + w, y + h), (255, 0, 0), 1)

plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.imshow(patente_rgb)
plt.title("Patente recortada")
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(patente_bin, cmap="gray", vmin=0, vmax=255)
plt.title(f"Umbralada T={T_elegido:.0f}")
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(patente_caracteres)
plt.title(f"Caracteres ({len(caracteres)})")
plt.xticks([])
plt.yticks([])

guardar_figura("p2_caracteres")
plt.show(block=True)

# --- Cada carácter por separado -----------------------------------

plt.figure(figsize=(10, 2))

for k, (x, y, w, h, area) in enumerate(caracteres):
    plt.subplot(1, len(caracteres), k + 1)
    plt.imshow(patente_bin[y:y + h, x:x + w], cmap="gray", vmin=0, vmax=255)
    plt.title(f"Carácter {k + 1}")
    plt.xticks([])
    plt.yticks([])

guardar_figura("p2_caracteres_individuales")
plt.show(block=True)






# --- Todas las imágenes -------------------------------------------


# Para cada imagen: primero probamos Otsu. Si no aparecen las 6 letras,
# probamos el umbral de respaldo
# Anotamos qué pasó con cada uno.

print(f"\nimagen | Otsu | letras Otsu | letras T={T_elegido} | resultado")
print("-------+------+-------------+--------------+-----------")

plt.figure(figsize=(16, 8))
con_otsu = 0
detectadas = 0

for n in range(1, 13):
    nombre = f"img{n:02d}"                                   # img01, img02, ..., img12
    img = cv2.imread(f"Patentes/{nombre}.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Otsu
    T_otsu, img_bin_otsu = cv2.threshold(img_gris, 0, 255, cv2.THRESH_OTSU)
    grupo_otsu = agrupar_letras(buscar_letras(img_bin_otsu))

    # 2) Umbral de respaldo 
    _, img_bin_resp = cv2.threshold(img_gris, T_elegido, 255, cv2.THRESH_BINARY)
    grupo_resp = agrupar_letras(buscar_letras(img_bin_resp))

    # 3) Nos quedamos con Otsu si funcionó; si no, con el de respaldo
    if len(grupo_otsu) == 6:
        T, grupo, resultado = T_otsu, grupo_otsu, "Otsu"
        con_otsu += 1
    elif len(grupo_resp) == 6:
        T, grupo, resultado = T_elegido, grupo_resp, "respaldo"
    else:
        T, grupo, resultado = None, [], "NO detectada"

    print(f"{nombre}  | {T_otsu:4.0f} | {len(grupo_otsu):11d} | {len(grupo_resp):12d} | {resultado}")

    plt.subplot(3, 4, n)
    plt.xticks([])
    plt.yticks([])

    if T is None:                                            # no se encontró la patente
        plt.imshow(img_rgb)
        plt.title(f"{nombre}: no detectada")
        continue

    detectadas += 1

    #rectángulo de la patente
    x_min = min(x for x, y, w, h, area in grupo)
    y_min = min(y for x, y, w, h, area in grupo)
    x_max = max(x + w for x, y, w, h, area in grupo)
    y_max = max(y + h for x, y, w, h, area in grupo)
    alto_letra = max(h for x, y, w, h, area in grupo)

    x0 = max(x_min - alto_letra, 0)
    x1 = x_max + alto_letra
    y0 = max(y_min - alto_letra, 0)
    y1 = y_max + alto_letra

    patente_rgb = img_rgb[y0:y1, x0:x1].copy()
    patente_gris = img_gris[y0:y1, x0:x1]

    #caracteres
    _, patente_bin = cv2.threshold(patente_gris, T, 255, cv2.THRESH_BINARY)
    caracteres = agrupar_letras(buscar_letras(patente_bin))

    for x, y, w, h, area in caracteres:
        cv2.rectangle(patente_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)

    plt.imshow(patente_rgb)
    plt.title(f"{nombre}: {resultado} (T={T:.0f})")

plt.suptitle(f"Patentes detectadas: {detectadas} de 12")
guardar_figura("p2_resumen")
plt.show(block=True)

print("\nSolo con Otsu:", con_otsu, "de 12")
print("Otsu + respaldo:", detectadas, "de 12")