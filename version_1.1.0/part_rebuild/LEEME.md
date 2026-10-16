# Part Rebuild 1.1.0

Deja limpios los modelos hechos con IA (Meshy y similares). En vez de "arreglar" una zona
fundida, la **sustituye por una pieza limpia** medida sobre el original. Todo en modo Objeto.

## Dónde
Vista 3D → tecla **N** → pestaña **Part Rebuild**.

## Flujo
1. **Marca la zona mala** → botón **Marcar zona** y pinta encima (sale en rosa).
   - Clic izquierdo pinta · **Shift** + clic borra marca · `F` cubo (marca hasta las aristas) · `B` pincel.
   - `[` `]` o Shift + rueda: tamaño · Ctrl+Z deshace · botón central gira · Esc o Intro: terminar.
2. **Arréglala**:
   - **Automático** (lo normal): prueba cilindro, muelle, revolución y extrusión y pone la que
     mejor encaja. Si ninguna encaja, te lo dice y te propone las otras salidas.
   - O elige tú la forma:
   | Botón | Para | Qué mide |
   |---|---|---|
   | **Muelle** | muelles de amortiguadores, resortes | eje, radio, hilo, espiras, sentido |
   | **Cilindro** | varillas, ejes, vástagos | eje, diámetro, largo |
   | **Revolución** | pomos, bujes, tapacubos, botellas | eje y perfil (radio a cada altura) |
   | **Extrusión** | lamas, soportes, perfiles, placas | dirección y contorno de la sección |
   | **Tubo** | jaulas, barras, cables | no hace falta marcar: clic sobre el tubo en cada extremo y esquina, Intro para crear |
   - Sale con el color de la zona y un **encaje ±mm** (lo que se separa del original).
   - **Quitar lo viejo** (activado): borra la zona marcada y tapa el hueco.
   - **Si no es ninguna forma** (bulto, zona rugosa, algo orgánico): **Suavizar zona**
     (Suave / Media / Fuerte). Solo toca lo marcado y no encoge; el borde de la zona no se mueve.
     Se puede pulsar varias veces; la marca se queda hasta que la quitas con la X.
3. **Ajusta** si hace falta: con la pieza nueva seleccionada salen sus deslizadores
   (radio, hilo, espiras, largo, extremos cerrados, vástago, giro…). Se puede mover y girar como cualquier objeto.

## Copiar lo bueno
- **Copiar del otro lado** (simetría): marca la zona mala y elige el eje (X, Y o Z de la pieza).
  Pone encima la copia reflejada del lado bueno. **Afinar plano** busca el plano exacto.
- **Clonar**: marca una pieza buena → **Guardar la buena**. Marca una mala → **Poner aquí**:
  coloca una copia encajada automáticamente. Repite con las demás. **Ajustar tamaño** si no son iguales.

## Para imprimir
- **Juntar** (recomendado): pieza + reconstrucciones en un solo objeto con sus colores.
  El laminador (Orca, Bambu, Prusa) funde los solapes al laminar. No se pierde detalle.
- **Fundir**: sólido cerrado por vóxeles. Las paredes más finas que el vóxel pueden
  desaparecer: usa **Alto** y revisa.

## Reglas
- El original no se toca: la primera vez se crea `<pieza>_color` y el original se oculta.
  Si la pieza ya es una copia de trabajo (de Color Retouch u otro complemento del taller), se usa esa.
- Lo que no tiene forma reconocible no se puede reconstruir: ahí, simetría, clonar o suavizar.

## Límites conocidos
- Revolución hace la pieza **lisa** alrededor del eje: pierde dientes, dibujo de neumático, tornillos.
- Extrusión usa solo el contorno exterior (sin agujeros interiores).
- En la simetría puede quedar una costura fina en el borde de la zona; **Juntar** o **Fundir** la cierran al imprimir.
