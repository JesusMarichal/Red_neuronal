# Red neuronal para predecir partidos de MLB

Red densa en Rust ([Burn](https://burn.dev)) entrenada con **resultados reales
de béisbol** descargados de la MLB Stats API (`statsapi.mlb.com`, pública y sin
API key).

## Qué hace

1. **Descarga** todos los partidos de temporada regular de varias temporadas
   (por defecto 2018-2026, saltando 2020 por ser una temporada de 60 partidos
   sin público). Los JSON crudos quedan cacheados en `data/raw/`.
2. **Construye 29 características** recorriendo los partidos en orden
   cronológico estricto. Para cada partido solo se usa lo ocurrido **antes** de
   ese partido: Elo con margen de victoria y arrastre entre temporadas, récord
   con shrinkage, carreras a favor/en contra por juego, forma de los últimos 10,
   expectativa pitagórica, días de descanso, forma reciente del pitcher abridor
   y el registro de cada equipo **en la condición en que juega** (el local por
   cómo rinde en casa, el visitante por cómo rinde fuera).
   Salvo la «forma» de los últimos 10, todos los acumulados cubren la temporada
   en curso **completa, desde el primer día**. El estado se actualiza *después* de emitir la fila, así que no hay
   fuga de datos.
3. **Entrena y valida hacia adelante en el tiempo** (walk-forward): cada mes se
   reentrena un modelo desde cero solo con los partidos anteriores y se predicen
   los partidos de ese mes.

## Interfaz web

```bash
cargo build --release
./target/release/Red_neuronal ui              # panel en http://127.0.0.1:8080, refresco cada 60 s
./target/release/Red_neuronal ui 15 9000 30   # 15 dias, puerto 9000, refresco cada 30 s
./target/release/Red_neuronal export          # solo genera dashboard.html, sin servidor
```

Se abre en una fecha concreta (Hoy por defecto) y se navega con las pestañas de día:
`Hoy · Mañana · sáb 12 · dom 13 …`, cada una con su conteo de partidos, cuántos tienen
pronóstico y cuántos están en juego. No se apilan todas las fechas en una sola lista.

Por partido se ve el logo, récord, Elo, AVG y ERA de ambos equipos; el abridor con su
efectividad, récord, WHIP, entradas y ponches; y un desplegable con los bateadores
ordenados por promedio y la rotación completa.

**Selección del día.** Encima de cada jornada hay un panel que filtra los partidos cuyo
favorito supera un umbral (55/60/65/70 %, seleccionable) y calcula: cuántos entran, la
probabilidad media, los aciertos esperados (suma de probabilidades), la probabilidad de
acertarlos **todos** (producto) y la de acertar al menos uno. Se contrasta siempre con el
acierto real del modelo en esa banda durante los últimos 30 días, y en las fechas ya
jugadas muestra cuántos acertó de verdad.

La pestaña **Resumen general** hace lo mismo pero agregando todas las fechas del rango:
suma de las probabilidades de cada favorito seleccionado, esa suma como tasa de acierto
esperada, la combinada de todo el rango (en formato «1 entre N» cuando es diminuta) y una
tabla día a día.

**Resultados.** En cuanto un partido empieza, la columna derecha pasa de probabilidad a
**marcador**, con el número del ganador resaltado. Al terminar se muestra el resultado
final completo (`TB 1 – 3 ATL`) junto a qué se había pronosticado y si acertó o falló.
Los partidos en juego llevan borde rojo y se actualizan solos.

**Sin abridor confirmado no hay porcentaje.** La probabilidad depende mucho de quién
lanza: no es lo mismo un abridor con efectividad de 2.50 que uno de 5.00. Si algún equipo
no ha anunciado el suyo, el partido aparece sin número y con la explicación de por qué,
en lugar de dar una cifra poco fiable. El pronóstico aparece solo cuando ambos equipos
confirman, normalmente con 3 o 4 días de antelación.

**Se actualiza solo cada minuto.** Un hilo en segundo plano vuelve a bajar el calendario
de los próximos días (marcadores en vivo y abridores recién anunciados) y la página lo
recoge sin recargar. Cuando un partido termina, entra al histórico y **la red se
reentrena con ese resultado**. El histórico completo se rebaja cada hora y las plantillas
cada media hora, así que el refresco por minuto es barato.

Internamente todo se calcula en precisión completa; el redondeo a dos decimales ocurre
solo al mostrarlo.

## Uso desde la terminal

```bash
./target/release/Red_neuronal fetch              # descarga y regenera el CSV
./target/release/Red_neuronal fetch 2023 2024    # solo esas temporadas
./target/release/Red_neuronal train              # corte temporal: pasado -> última temporada
./target/release/Red_neuronal backtest 2024-04-01 # walk-forward mes a mes
./target/release/Red_neuronal predict 2026-08-15 # jornada pasada, partido a partido
./target/release/Red_neuronal                    # train + backtest + predict
```

El dataset procesado se guarda en `baseball_data.csv` (una fila por partido:
metadatos + las 29 características + el resultado real).

## Estructura

| Archivo | Rol |
|---|---|
| `src/mlb.rs` | Cliente de la MLB Stats API y cache en disco |
| `src/features.rs` | Estado cronológico y cálculo de las 29 características |
| `src/dataset.rs` | CSV y normalizador z-score (ajustado solo con entrenamiento) |
| `src/model.rs` | MLP 29 → 32 → 16 → 2 con ReLU y dropout |
| `src/training.rs` | Mini-lotes, Adam con weight decay, parada temprana, métricas |
| `src/backtest.rs` | Walk-forward, baselines, calibración, demo por jornada |
| `src/stats.rs` | Plantillas, bateadores y lanzadores de la temporada en curso |
| `src/live.rs` | Motor en vivo: refresco, reentrenamiento y payload JSON |
| `src/ui.rs` + `src/dashboard.html` | Servidor HTTP local con refresco en segundo plano |

## Sobre los resultados

Predecir béisbol es difícil: es el deporte con más azar de las grandes ligas y
las casas de apuestas rondan el 57-58 % de acierto. Por eso el modelo se compara
siempre contra tres baselines (Elo, mejor récord, siempre el local) y se reporta
también **logloss**, **Brier** y **calibración**: acertar el ganador importa
menos que dar una probabilidad honesta.
