# Red neuronal para predecir partidos de MLB

Red densa en Rust ([Burn](https://burn.dev)) entrenada con **resultados reales
de béisbol** descargados de la MLB Stats API (`statsapi.mlb.com`, pública y sin
API key).

## Qué hace

1. **Descarga** todos los partidos de temporada regular de varias temporadas
   (por defecto 2018-2026, saltando 2020 por ser una temporada de 60 partidos
   sin público). Los JSON crudos quedan cacheados en `data/raw/`.
2. **Construye 25 características** recorriendo los partidos en orden
   cronológico estricto. Para cada partido solo se usa lo ocurrido **antes** de
   ese partido: Elo con margen de victoria y arrastre entre temporadas, récord
   con shrinkage, carreras a favor/en contra por juego, forma de los últimos 10,
   expectativa pitagórica, días de descanso y forma reciente del pitcher
   abridor. El estado se actualiza *después* de emitir la fila, así que no hay
   fuga de datos.
3. **Entrena y valida hacia adelante en el tiempo** (walk-forward): cada mes se
   reentrena un modelo desde cero solo con los partidos anteriores y se predicen
   los partidos de ese mes.

## Interfaz web

```bash
cargo build --release
./target/release/Red_neuronal ui           # panel en http://127.0.0.1:8080 (abre el navegador)
./target/release/Red_neuronal ui 15 9000   # 15 dias por delante, en el puerto 9000
./target/release/Red_neuronal export       # solo genera dashboard.html, sin servidor
```

El panel muestra, fecha por fecha, cada partido programado con:

- logo, récord, Elo, AVG y ERA de ambos equipos;
- probabilidad de victoria de cada uno, con la referencia del modelo Elo al lado;
- el abridor de cada equipo con su efectividad, récord, WHIP, entradas y ponches;
- desplegable con la plantilla completa: bateadores ordenados por promedio (AVG, OBP,
  OPS, HR, CI) y la rotación de abridores;
- tabla de los 30 equipos ordenada por Elo.

**Fechas confirmadas.** Los equipos anuncian su abridor con pocos días de antelación, así
que solo una parte de los partidos tiene ambos lanzadores confirmados. Esos llevan una
etiqueta verde; el resto usa el promedio de liga en lugar del lanzador real y es menos
fiable. El botón «Solo abridores confirmados» filtra los firmes. Los partidos del día que
ya terminaron muestran el marcador real y si el pronóstico acertó.

Las probabilidades salen del estado de los equipos **a día de hoy**: no se simulan los
partidos intermedios, así que cuanto más lejana la fecha, más provisional el número.
Internamente todo se calcula en precisión completa; el redondeo a dos decimales ocurre
solo al mostrarlo.

Los datos de la temporada en curso (calendario, plantillas, estadísticas) se refrescan
solos cada pocas horas; las temporadas cerradas se cachean para siempre en `data/raw/`.

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
metadatos + las 25 características + el resultado real).

## Estructura

| Archivo | Rol |
|---|---|
| `src/mlb.rs` | Cliente de la MLB Stats API y cache en disco |
| `src/features.rs` | Estado cronológico y cálculo de las 25 características |
| `src/dataset.rs` | CSV y normalizador z-score (ajustado solo con entrenamiento) |
| `src/model.rs` | MLP 25 → 32 → 16 → 2 con ReLU y dropout |
| `src/training.rs` | Mini-lotes, Adam con weight decay, parada temprana, métricas |
| `src/backtest.rs` | Walk-forward, baselines, calibración, demo por jornada |
| `src/stats.rs` | Plantillas, bateadores y lanzadores de la temporada en curso |
| `src/live.rs` | Predicción de partidos futuros y payload JSON del panel |
| `src/ui.rs` + `src/dashboard.html` | Interfaz web y servidor HTTP local |

## Sobre los resultados

Predecir béisbol es difícil: es el deporte con más azar de las grandes ligas y
las casas de apuestas rondan el 57-58 % de acierto. Por eso el modelo se compara
siempre contra tres baselines (Elo, mejor récord, siempre el local) y se reporta
también **logloss**, **Brier** y **calibración**: acertar el ganador importa
menos que dar una probabilidad honesta.
