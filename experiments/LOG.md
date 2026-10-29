# S6E10 — registro de experimentos

Validación: `StratifiedKFold(5, shuffle=True, random_state=42)` sobre `train.csv` en orden de id (`folds.npy` de cada
kernel). Métrica: ROC AUC fuera de pliegue (OOF) sobre las 699.635 filas de la competición. Las filas del dataset
original, cuando se usan, solo entran en el entrenamiento de cada pliegue; nunca en validación.
Cómputo: kernels privados de Kaggle (JEV 0,98). Tabla máquina: `runs.csv` (id, fecha, hipótesis, config, commit,
semilla, métrica, decisión). Los resultados negativos también se registran.

## Hipótesis

- **H0 (control profesional):** GBDT bien ajustados (CatBoost, LightGBM, XGBoost) con Optuna y blend lineal OOF.
- **H1 (tecnología propia):** Darwin-Cage, la búsqueda evolutiva de programas de agrupación sobre el residuo OOF
  (repo `Agnuxo1/s6e9-honest-ceiling`), encuentra estructura confirmada en el pliegue reservado que sube el AUC OOF
  del blend con 3 semillas.
- **H2 (datos externos):** añadir las 129.880 filas del dataset original CC0 al entrenamiento de cada pliegue sube el AUC.
- **H3 (representación):** tratar las valoraciones 0–5, o todas las columnas, como categóricas en CatBoost sube el AUC.

## Entradas

### E001 — CatBoost base (H0) · 28-10 12:14 UTC · commit d3bdc62 · kernel `s6e10-e001-cat-base`
CatBoost GPU, 4 categóricas, depth 7, lr 0,06, parada temprana Logloss (od 300), semilla 0.
OOF AUC **0,958327** (pliegues 0,95738–0,95918), ~1.900 iteraciones, 6,5 min. Envío 57736176: pública **0,95798**.
Decisión: baseline y primer envío. Referencia pública (goodpjw2008, mismos pliegues): XGB crudo 0,958763.

### Investigación pública (28-10 13:30 UTC)
Notebooks de goodpjw2008 (serie Parts 1–14, `StratifiedKFold(5, 42)` como nosotros): Flight Distance como id de
ruta (TE, recuentos, perfil) +0,00160 en XGB; predicción de un modelo entrenado en el original +0,00072; CatBoost
con numéricas y valoraciones categóricas 0,961064; stacks LR anidados ~0,9619 CV / 0,9616 LB. Biblioteca pública
`cdeotte/s6e10-124-public-oof-and-test-predictions` (124 miembros, alineados por id; mejor individual OOF 0,961744;
licencia «unknown»: se usa, no se redistribuye). No hay filas copiadas del original (17 coincidencias exactas de 700k);
todos los valores numéricos pertenecen al conjunto de valores del original.

## Sesión 29-10 (Claude, agente s6e10)

### E002–E014 (kernels del 28-10, descargados 29-10 05:15 UTC) · tabla en runs.csv
| id | modelo | variante | OOF AUC |
|---|---|---|---|
| E002 | CatBoost | valoraciones categóricas | 0,958455 |
| E003 | CatBoost | todas categóricas | 0,960314 |
| E004/E005 | CatBoost | + filas del original (sin/con marca) | 0,958165 / ver runs.csv |
| E006/E007/E008 | LightGBM | base / + original / + ingeniería | 0,958983 / 0,958769 / 0,958851 |
| E009/E010 | XGBoost | base / + original | 0,958851 / 0,958694 |
| E011 | XGBoost | ruta + og + TE fold-safe | 0,960872 |
| E012 | LightGBM | ing. + ruta + og + TE | 0,960834 |
| E013 | CatBoost | todas cat + og | 0,960916 |
| E014 | CatBoost | todas cat + ruta + og | **0,960984** |
Decisiones: **H2 rechazada** (las filas del original restan 0,0001–0,0002 en los tres GBDT). **H3 confirmada** (todas
categóricas +0,0020 en CatBoost; Flight Distance como id de ruta +0,0020 en XGB/LGBM).

### D001 — Darwin-Cage sobre el residuo de E001 (H1, 28-10)
35 min, 7.547 programas, 549 generaciones, umbral 0,01134 (z=3), 3.186 confirmados (FP esperados 4,7).
Familia ganadora: clave Flight_Distance × Type_of_Travel (4.594 claves); pliegue reservado 0,95858 → 0,95959 (+0,0010).
Lectura: Darwin redescubre sin guía la «ruta» que E011–E014 ya codifican a mano. Decisión: probar sobre el residuo del
blend (D002), que ya contiene la ruta.

### B001 — blend OOF de los 14 GBDT propios · 29-10 05:10 UTC · local `tools/blend.py`
Hill climbing LOFO 0,961226 (E011+E013+E014 a partes iguales); stack LR sobre logits LOFO **0,961244** (mejor
individual 0,960984). Envío 57759629: pública **0,96083**. Decisión: control profesional H0 de referencia.

### S001 — stack LR anidado con la biblioteca pública cdeotte124 · kernel `s6e10-s001-stack` · 29-10 05:20 UTC
138 miembros (14 propios + 124 públicos; mejor individual público nj_01 0,961744). LR sobre logits, CV anidada con
los mismos 5 pliegues. Solo propios: 0,961246–0,961248 (C 0,001–0,03). Propios+públicos: C=0,001 0,962061;
**C=0,003 0,962066** (pliegues 0,96217/0,96127/0,96267/0,96227/0,96197); C=0,03 0,962047. Ganancia de los miembros
públicos sobre el blend propio: +0,00082. Enviado (lr_all_C0.003).
### D002 — Darwin-Cage sobre el residuo de B001 (H1 sobre el blend GBDT) · semillas 2027/2028/2029 · 29-10 05:17–05:48
35 min por semilla; 7.674 / 8.088 / 7.174 programas evaluados; umbral 0,01134 (z=3). Confirmados: **1 / 0 / 1**, el
mismo programa (Flight_Distance÷100 × Type_of_Travel × Arrival_Delay÷50 × Age mod 11, confirmación 0,01197, apenas
sobre el umbral). Aplicado fold-safe sobre el blend: OOF 0,961234 → 0,960636 (−0,0006), rechazado en las dos semillas
que lo tienen. Reaplicación con calibración Platt (`tools/darwin_reapply.py`, evita los empates de la isotónica):
mismo resultado. **H1 rechazada sobre el residuo del blend GBDT (3/3 semillas sin ganancia).** Lectura: el residuo de
B001 ya no contiene la estructura de ruta que D001 encontró sobre E001; las vistas ruta/og de E011–E014 la absorben.
### D003 — Darwin-Cage sobre el residuo de S001 (stack final) · semillas 3027/3028/3029 · 29-10 05:48–06:44
13.864 / 7.412 / 7.008 programas evaluados (35 min cada una), umbral 0,01134. Confirmados: **0 / 0 / 0**. Sin
programas que aplicar: el stack con calibración Platt queda en 0,962063 (−0,000004 por la calibración por pliegue).
**H1 rechazada también sobre el stack final.** Por la regla de JEV (answer-002, 0,56: ganancia positiva en las 3
semillas y mayor que su desviación), Darwin-Cage NO entra en los envíos finales. Su aportación medida es diagnóstica:
sobre un modelo único (D001) redescubre sin guía la clave de ruta (+0,0010 en el pliegue reservado) y certifica que el
residuo del blend/stack no conserva estructura de agrupación detectable con z=3.
