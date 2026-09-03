# Degradación procedural reproducible de un laberinto

Este proyecto genera imágenes raster que se parecen a una impresión fotografiada o escaneada con pequeñas imperfecciones. Todas las operaciones aleatorias usan `numpy.random.default_rng(seed)`. No se usan modelos generativos, inpainting, vectorización, OCR, SVG ni servicios externos.

## Instalación y ejecución

```bash
python -m pip install -r requirements.txt
python degrade_maze.py maze-10x10-kids-1788448203980.png \
  --output-dir output --seed 423 --debug --pdf
```

Opciones útiles:

```text
--roi x,y,w,h       ROI manual cuando la detección automática no es fiable
--levels all         all o lista: subtle,low,medium,strong,max_readable
--pdf               crea el PDF A4 image-only
--debug             guarda máscaras, desplazamientos y proxies de recuperación
--no-perspective    desactiva la perspectiva global leve
```

## Diseño de seguridad

La imagen se analiza en RGB, escala de grises y HSV. Se detecta el ROI, se extrae una máscara aproximada de paredes, se estima su grosor mediante `skeletonize` y `distanceTransform`, y se construyen `WALL_CORE`, `WALL_EDGE` y `CORRIDOR_CORE`. El campo elástico se rechaza si el determinante Jacobiano no supera 0.60; además, la máscara deformada se compara con la original usando componentes, Euler, recorte y continuidad del núcleo.

La textura, iluminación, pseudo-gaps y distractores sólo actúan sobre regiones permitidas. El núcleo de pared se vuelve a imponer al final como guardrail visual; los marcadores verdes se preservan. Cada variante tiene hasta diez intentos con amplitud decreciente. Un fallo queda registrado en `run_config.json` y no se presenta como validación PASS.

El PDF se compone a 300 DPI como una página raster de 2480×3508 y ReportLab inserta únicamente esa imagen completa. No hay texto vectorial, trazados, OCR ni reconstrucción de paredes dentro del PDF.

## Salidas

Se generan PNG/JPEG por nivel, `contact_sheet.png`, `debug_sheet.png`, métricas en `run_config.json`, y, con `--pdf`, `maze_recommended_A4.png` y `maze_recommended_A4.pdf`. La selección automática prefiere `medium`, luego `strong`, siempre que pasen todas las validaciones; `max_readable` es experimental.

Los proxies de recuperación (`debug/recovery`) son únicamente una comprobación heurística de cuánto ayuda el preprocesamiento clásico. No representan una garantía de seguridad ni de imposibilidad de lectura automática: el objetivo es aumentar la dificultad de extracción manteniendo legibilidad humana y la topología del ejercicio.

## Tests

```bash
pytest -q
```
