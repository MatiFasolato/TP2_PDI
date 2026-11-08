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

# --- Paso 1: reducir, suavizar y calcular gradiente --------------------

# Reducción de escala (2.1). INTER_AREA promedia píxeles y elimina el grano.
escala = 0.25
img_ch = cv2.resize(img_gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
print("Tamaño reducido:", img_ch.shape)

# Suavizado leve (2.4)
img_suave = cv2.GaussianBlur(img_ch, (5, 5), 0)

# Gradiente con Sobel (3.2): derivadas en x e y y magnitud
gx = cv2.Sobel(img_suave, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img_suave, cv2.CV_64F, 0, 1, ksize=3)
magnitud = np.sqrt(gx**2 + gy**2)

# Normalizo a 0-255 para visualizar
magnitud_u8 = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(15, 5))

ax1 = plt.subplot(131)
plt.imshow(img_suave, cmap="gray")
plt.title("Reducida y suavizada")
plt.axis("off")

plt.subplot(132, sharex=ax1, sharey=ax1)
plt.imshow(magnitud_u8, cmap="gray")
plt.title("Magnitud del gradiente")
plt.axis("off")

plt.subplot(133)
plt.hist(magnitud.ravel(), bins=256)
plt.yscale("log")
plt.title("Histograma de la magnitud (escala log)")
plt.xlabel("Magnitud")

plt.tight_layout()
plt.show()


# --- Paso 2: umbralar la magnitud del gradiente ------------------------

# Referencia: Otsu sobre el gradiente
T_otsu, bordes_otsu = cv2.threshold(magnitud_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Umbral Otsu sobre el gradiente:", T_otsu)

# Percentiles para tener una idea de la escala de valores
for p in (50, 90, 95, 99):
    print(f"Percentil {p}: {np.percentile(magnitud_u8, p):.1f}")

# Tres umbrales fijos bajos
umbrales = [10, 20, 30]
plt.figure(figsize=(15, 9))

ax1 = plt.subplot(221)
plt.imshow(bordes_otsu, cmap="gray")
plt.title(f"Otsu T={T_otsu:.0f}")
plt.axis("off")

for idx, T in enumerate(umbrales):
    _, bordes_T = cv2.threshold(magnitud_u8, T, 255, cv2.THRESH_BINARY)
    plt.subplot(2, 2, idx + 2, sharex=ax1, sharey=ax1)
    plt.imshow(bordes_T, cmap="gray")
    plt.title(f"Umbral fijo T={T}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# --- Paso 3: limpiar, cerrar y rellenar --------------------------------

# Bordes con el umbral elegido
_, bordes = cv2.threshold(magnitud_u8, 20, 255, cv2.THRESH_BINARY)

# 1) Elimino ruido suelto por área de componente conectada (3.3)
n, labels, stats, _ = cv2.connectedComponentsWithStats(bordes, connectivity=8)
bordes_limpios = np.zeros_like(bordes)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 50:      # umbral a ajustar
        bordes_limpios[labels == i] = 255
print("Componentes antes de limpiar:", n - 1)

def cerrar_y_rellenar(bordes, k):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 2) Dilato para cerrar los cortes que quedan en los contornos
    cerrado = cv2.dilate(bordes, kernel)

    # 3) Relleno desde el fondo: lo que no se inunda es interior de un objeto
    inundado = cerrado.copy()
    aux = np.zeros((cerrado.shape[0] + 2, cerrado.shape[1] + 2), np.uint8)
    cv2.floodFill(inundado, aux, (0, 0), 255)
    huecos = cv2.bitwise_not(inundado)
    relleno = cv2.bitwise_or(cerrado, huecos)

    # Erosiono con el mismo kernel para recuperar el tamaño
    return cv2.erode(relleno, kernel)

plt.figure(figsize=(15, 5))
ax1 = plt.subplot(131)
plt.imshow(bordes_limpios, cmap="gray")
plt.title("Bordes limpios (T=20)")
plt.axis("off")

for idx, k in enumerate((7, 11)):
    mascara_k = cerrar_y_rellenar(bordes_limpios, k)
    n_obj, _, _, _ = cv2.connectedComponentsWithStats(mascara_k, connectivity=8)
    print(f"Kernel {k}: {n_obj - 1} componentes")

    plt.subplot(1, 3, idx + 2, sharex=ax1, sharey=ax1)
    plt.imshow(mascara_k, cmap="gray")
    plt.title(f"Relleno kernel {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()


# --- Paso 4: envolvente convexa por componente (máscara kernel 7) ------

mascara_7 = cerrar_y_rellenar(bordes_limpios, 7)

n, labels, stats, cents = cv2.connectedComponentsWithStats(mascara_7, connectivity=8)
print("Componentes en mascara_7:", n - 1)

# Áreas ordenadas, para detectar fragmentos o uniones
areas = sorted(stats[1:, cv2.CC_STAT_AREA])
print("Áreas ordenadas:", areas)

# Envolvente convexa de cada componente, por separado
mascara_hull = np.zeros_like(mascara_7)
for i in range(1, n):
    comp = np.uint8(labels == i) * 255
    contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contornos, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(mascara_hull, [hull], -1, 255, thickness=-1)

# Cuento los objetos finales
n_final, _, _, _ = cv2.connectedComponentsWithStats(mascara_hull, connectivity=8)
print("Objetos tras la envolvente:", n_final - 1, "(esperado: 19)")

# Contornos de la máscara final sobre la imagen reducida, para verificar
contornos_f, _ = cv2.findContours(mascara_hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = cv2.cvtColor(img_ch, cv2.COLOR_GRAY2RGB)
cv2.drawContours(overlay, contornos_f, -1, (255, 0, 0), 2)

plt.figure(figsize=(15, 5))

ax1 = plt.subplot(131)
plt.imshow(mascara_7, cmap="gray")
plt.title("Kernel 7")
plt.axis("off")

plt.subplot(132, sharex=ax1, sharey=ax1)
plt.imshow(mascara_hull, cmap="gray")
plt.title("Con envolvente convexa")
plt.axis("off")

plt.subplot(133, sharex=ax1, sharey=ax1)
plt.imshow(overlay)
plt.title("Contornos sobre la imagen")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Paso 5: descriptores geométricos de cada objeto -------------------

n, labels, stats, cents = cv2.connectedComponentsWithStats(mascara_hull, connectivity=8)

objetos = []
for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]

    comp = np.uint8(labels == i) * 255
    contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(cnt, True)

    circularidad = 4 * np.pi * area / perimetro**2

    objetos.append({
        "id": i,
        "area": area,
        "perimetro": perimetro,
        "circularidad": circularidad,
        "centroide": cents[i],
    })

# Tabla ordenada por circularidad
print(f"{'id':>3} {'area':>7} {'perim':>8} {'circ':>6}")
for o in sorted(objetos, key=lambda o: o["circularidad"]):
    print(f"{o['id']:>3} {o['area']:>7} {o['perimetro']:>8.1f} {o['circularidad']:>6.3f}")

# Imagen con el número de cada objeto
plt.figure(figsize=(10, 7))
plt.imshow(img_ch, cmap="gray")
for o in objetos:
    cx, cy = o["centroide"]
    plt.text(cx, cy, str(o["id"]), color="red", fontsize=12,
             fontweight="bold", ha="center", va="center")
plt.title("Número de cada objeto")
plt.axis("off")
plt.show()


# --- Paso 6: dados vs monedas, y tipos de moneda con k-means ------------

# 1) Dados vs monedas por circularidad
UMBRAL_CIRC = 0.87
dados = [o for o in objetos if o["circularidad"] < UMBRAL_CIRC]
monedas = [o for o in objetos if o["circularidad"] >= UMBRAL_CIRC]
print("Dados:", len(dados), "| Monedas:", len(monedas))

# 2) K-means (k=3) sobre el área de las monedas
areas = np.float32([o["area"] for o in monedas]).reshape(-1, 1)
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
_, etiquetas, centros = cv2.kmeans(areas, 3, None, criterio, 10, cv2.KMEANS_PP_CENTERS)

# Ordeno los grupos por tamaño: 0 = chica, 1 = mediana, 2 = grande
orden = np.argsort(centros.flatten())
rank = {int(c): r for r, c in enumerate(orden)}
for o, e in zip(monedas, etiquetas.flatten()):
    o["tipo"] = rank[int(e)]

nombres = ["Chica", "Mediana", "Grande"]
print("\nCentros de área:", np.sort(centros.flatten()))
for t in range(3):
    grupo = [o["area"] for o in monedas if o["tipo"] == t]
    print(f"{nombres[t]}: {len(grupo)} monedas | área {min(grupo)} - {max(grupo)}")

# 3) Visualización: un color por categoría
colores = {"dado": (255, 255, 255), 0: (255, 0, 0), 1: (0, 200, 0), 2: (0, 0, 255)}
salida = np.zeros((*mascara_hull.shape, 3), np.uint8)
for o in dados:
    salida[labels == o["id"]] = colores["dado"]
for o in monedas:
    salida[labels == o["id"]] = colores[o["tipo"]]

plt.figure(figsize=(10, 7))
plt.imshow(salida)
plt.title(f"Dados (blanco): {len(dados)} | Chicas (rojo): "
          f"{sum(o['tipo']==0 for o in monedas)} | Medianas (verde): "
          f"{sum(o['tipo']==1 for o in monedas)} | Grandes (azul): "
          f"{sum(o['tipo']==2 for o in monedas)}")
plt.axis("off")
plt.show()



# --- Paso 7: recorte de cada dado y detección de puntos ----------------

factor = int(1 / escala)   # 4
margen = 3

for d in dados:
    # Bounding box en la imagen reducida
    x = stats[d["id"], cv2.CC_STAT_LEFT]
    y = stats[d["id"], cv2.CC_STAT_TOP]
    w = stats[d["id"], cv2.CC_STAT_WIDTH]
    h = stats[d["id"], cv2.CC_STAT_HEIGHT]
    x0, y0 = max(0, x - margen), max(0, y - margen)
    x1, y1 = x + w + margen, y + h + margen

    # Recorte en la imagen original (mayor resolución)
    crop = img_gris[y0*factor:y1*factor, x0*factor:x1*factor]

    # Máscara del dado, llevada al tamaño del recorte y erosionada
    # para no tomar el borde del dado como "punto"
    mask_crop = np.uint8(labels[y0:y1, x0:x1] == d["id"])
    mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_int = cv2.erode(mask_crop, k)

    # Umbralado de Otsu usando solo los píxeles del dado
    T, _ = cv2.threshold(crop[mask_int == 1].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invierto: puntos (oscuros) = 1, dentro del dado
    puntos = np.uint8((crop < T) & (mask_int == 1))

    # Componentes conectadas (3.3) y descriptores de cada una
    n_p, lab_p, st_p, ce_p = cv2.connectedComponentsWithStats(puntos, connectivity=8)

    print(f"\nDado id {d['id']} | Otsu T = {T:.0f} | componentes oscuras: {n_p - 1}")
    print(f"{'comp':>5} {'area':>6} {'ancho':>6} {'alto':>5} {'circ':>6}")
    for i in range(1, n_p):
        a = st_p[i, cv2.CC_STAT_AREA]
        cm = np.uint8(lab_p == i)
        cn, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p = cv2.arcLength(max(cn, key=cv2.contourArea), True)
        c = 4 * np.pi * a / p**2 if p > 0 else 0
        print(f"{i:>5} {a:>6} {st_p[i, cv2.CC_STAT_WIDTH]:>6} "
              f"{st_p[i, cv2.CC_STAT_HEIGHT]:>5} {c:>6.3f}")

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(121)
    plt.imshow(crop, cmap="gray")
    plt.title(f"Dado id {d['id']}")
    plt.axis("off")

    plt.subplot(122, sharex=ax1, sharey=ax1)
    plt.imshow(puntos, cmap="gray")
    plt.title("Píxeles oscuros dentro del dado")
    plt.axis("off")
    plt.show()
    
    
    # --- Paso 8: contar puntos de la cara superior --------------------------

AREA_MIN_PUNTO = 1000
CIRC_MIN_PUNTO = 0.7

for d in dados:
    x = stats[d["id"], cv2.CC_STAT_LEFT]
    y = stats[d["id"], cv2.CC_STAT_TOP]
    w = stats[d["id"], cv2.CC_STAT_WIDTH]
    h = stats[d["id"], cv2.CC_STAT_HEIGHT]
    x0, y0 = max(0, x - margen), max(0, y - margen)
    x1, y1 = x + w + margen, y + h + margen

    crop = img_gris[y0*factor:y1*factor, x0*factor:x1*factor]

    mask_crop = np.uint8(labels[y0:y1, x0:x1] == d["id"])
    mask_crop = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask_int = cv2.erode(mask_crop, k)

    T, _ = cv2.threshold(crop[mask_int == 1].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    puntos = np.uint8((crop < T) & (mask_int == 1))

    n_p, lab_p, st_p, ce_p = cv2.connectedComponentsWithStats(puntos, connectivity=8)

    # Filtro por área y circularidad
    validos = []
    for i in range(1, n_p):
        a = st_p[i, cv2.CC_STAT_AREA]
        if a < AREA_MIN_PUNTO:
            continue
        cm = np.uint8(lab_p == i)
        cn, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p = cv2.arcLength(max(cn, key=cv2.contourArea), True)
        c = 4 * np.pi * a / p**2 if p > 0 else 0
        if c >= CIRC_MIN_PUNTO:
            validos.append(i)

    d["valor"] = len(validos)
    print(f"Dado id {d['id']}: cara superior = {d['valor']}")

    # Verificación visual: marco los puntos contados
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    for i in validos:
        cx, cy = ce_p[i]
        cv2.circle(vis, (int(cx), int(cy)), 40, (255, 0, 0), 5)

    plt.figure(figsize=(5, 5))
    plt.imshow(vis)
    plt.title(f"Dado id {d['id']}: valor = {d['valor']}")
    plt.axis("off")
    plt.show()

# Resumen del punto C
valores = [d["valor"] for d in dados]
print("\nCantidad de dados:", len(dados))
print("Valores de las caras superiores:", valores)
print("Suma total:", sum(valores))


# --- Paso 9: imagen final con etiquetas -------------------------------

# Asignación de denominación por tamaño (k-means ordenado: 0 chica, 1 mediana, 2 grande)
DENOMINACION = {0: "Moneda: $0.10", 1: "Moneda: $1", 2: "Moneda: $0.50"}
COLORES = {
    "dado": "#ff8c00",
    0: "#1f77b4",   # $0.10
    1: "#2ca02c",   # $1
    2: "#d62728",   # $0.50
}

# Imagen en color al mismo tamaño que la máscara
img_color_ch = cv2.resize(img_rgb, (img_ch.shape[1], img_ch.shape[0]),
                          interpolation=cv2.INTER_AREA)

fig, ax = plt.subplots(figsize=(14, 9))
ax.imshow(img_color_ch)

for o in objetos:
    if "tipo" in o:                      # es moneda
        texto = DENOMINACION[o["tipo"]]
        color = COLORES[o["tipo"]]
    else:                                # es dado
        texto = f"Dado: {o['valor']}"
        color = COLORES["dado"]

    # Contorno del objeto
    comp = np.uint8(labels == o["id"]) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
    cnt = np.vstack([cnt, cnt[:1]])      # cierro el contorno
    ax.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=2)

    # Etiqueta arriba del objeto
    cx, _ = o["centroide"]
    y_top = stats[o["id"], cv2.CC_STAT_TOP]
    ax.text(cx, y_top - 6, texto, color="white", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.9, pad=2, edgecolor="none"))

n_dados = len(dados)
n_10 = sum(o["tipo"] == 0 for o in monedas)
n_1 = sum(o["tipo"] == 1 for o in monedas)
n_50 = sum(o["tipo"] == 2 for o in monedas)
ax.set_title(f"Dados: {n_dados} | $0.10: {n_10} | $1: {n_1} | $0.50: {n_50}")
ax.axis("off")

fig.savefig("resultado_final.png", dpi=150, bbox_inches="tight")
plt.show()