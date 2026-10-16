# Part Rebuild

Complemento de Blender (4.2 – 5.x) para dejar limpios los modelos generados por IA (Meshy y similares):
sustituye las zonas fundidas por piezas limpias medidas sobre el original. Todo en modo Objeto.

- **Un clic**: «Arreglar esta pieza» marca la pieza sola, elige la forma y la sustituye.
- **Marca que se corrige sola**: pinta a ojo; quita lo que sobra y rellena lo que falta.
- **Quita solo lo chorreado**: lo de alrededor (soportes, anclajes) se conserva.
- **Automático**: elige solo la forma que mejor encaja; si no es ninguna, **Suavizar zona**.
- **Formas**: muelle, cilindro, tubo por clics, revolución y extrusión.
- **Copiar lo bueno**: simetría (lado bueno sobre el malo) y clonar una pieza buena sobre las malas.

Instrucciones: [part_rebuild/LEEME.md](part_rebuild/LEEME.md).

Instalar: `python crear_zip.py` → Blender → Preferencias → Complementos → Instalar desde disco → `dist/part_rebuild-1.3.0.zip`.

## Licencia

GPL-3.0-or-later (ver [LICENSE](LICENSE)). Gratis para usar, modificar y compartir; si redistribuyes una versión modificada, publica también su código. Es la licencia que exige extensions.blender.org.
