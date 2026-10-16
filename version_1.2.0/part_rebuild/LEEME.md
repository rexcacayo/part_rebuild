# Part Rebuild 1.2.0

Deja limpios los modelos hechos con IA (Meshy y similares). En vez de "arreglar" una zona
fundida, la **sustituye por una pieza limpia** medida sobre el original. Todo en modo Objeto.

## Dónde
Vista 3D → tecla **N** → pestaña **Part Rebuild**.

## Lo más rápido: «Arreglar esta pieza (1 clic)»
Pulsa el botón grande y haz **un clic** sobre la pieza mala (muelle, varilla, tubo, amortiguador…).
El plugin la marca entera solo, elige la forma, pone la pieza limpia y quita la vieja.
Si no le gusta, **Ctrl+Z** y vuelta a empezar.

## Paso a paso
1. **Marca la pieza**
   - **Clic en la pieza**: un clic sobre una pieza redonda y se marca entera (crece por la
     pieza y se para donde cambia). Clic en otra para añadirla. Esc para terminar.
   - **Pintar**: pinta a ojo por encima. Con **Corregir la marca sola** (activado), al terminar
     quita los trozos sueltos y lo que sobra, rellena huecos y alarga hasta el final de la pieza.
     - Clic izquierdo pinta · **Shift** + clic borra marca · `[` `]` tamaño · Ctrl+Z deshace · Esc termina.
   - Junto a «Marcadas»: botón para **corregir la marca** otra vez y **X** para quitarla.
2. **Arréglala**
   - **Automático** (lo normal): prueba cilindro, muelle, revolución y extrusión y pone la que
     mejor encaja. Si ninguna encaja, lo dice y propone las otras salidas.
   - O elige tú la forma:
   | Botón | Para | Qué mide |
   |---|---|---|
   | **Muelle** | muelles de amortiguadores, resortes | eje, radio, hilo, espiras, sentido |
   | **Cilindro** | varillas, ejes, vástagos | eje, diámetro, largo |
   | **Revolución** | pomos, bujes, tapacubos, botellas | eje y perfil (radio a cada altura) |
   | **Extrusión** | lamas, soportes, perfiles, placas | dirección y contorno de la sección |
   | **Tubo** | jaulas, barras, cables | sin marcar: clic sobre el tubo en cada extremo y esquina, Intro para crear |
   - Sale con el color de la zona y un **encaje ±mm** (lo que se separa del original).
   - **Quitar lo viejo** (activado): borra la zona marcada y tapa el hueco.
   - **Si no es ninguna forma** (bulto, zona rugosa, algo orgánico): **Suavizar zona**
     (Suave / Media / Fuerte). Solo toca lo marcado y no encoge; el borde no se mueve.
3. **Ajusta** si hace falta: con la pieza nueva seleccionada salen sus deslizadores
   (radio, hilo, espiras, largo, extremos cerrados, vástago, giro…). Se mueve y gira como cualquier objeto.

## Copiar lo bueno
- **Copiar del otro lado** (simetría): marca la zona mala y elige el eje (X, Y o Z de la pieza).
  Pone encima la copia reflejada del lado bueno. **Afinar plano** busca el plano exacto.
- **Clonar**: marca una pieza buena → **Guardar la buena**. Marca una mala → **Poner aquí**:
  coloca una copia encajada automáticamente. Repite con las demás. **Ajustar tamaño** si no son iguales.

## Reglas
- El original no se toca: la primera vez se crea `<pieza>_color` y el original se oculta.
  Si la pieza ya es una copia de trabajo (de Color Retouch u otro complemento del taller), se usa esa.
- Lo que no tiene forma reconocible no se puede reconstruir: ahí, simetría, clonar o suavizar.

## Límites conocidos
- Revolución hace la pieza **lisa** alrededor del eje: pierde dientes, dibujo de neumático, tornillos.
- Extrusión usa solo el contorno exterior (sin agujeros interiores).
- En la simetría puede quedar una costura fina en el borde de la zona.
- El clic automático es para piezas redondas/alargadas. Lo demás: pintar (se corrige solo), simetría o clonar.
