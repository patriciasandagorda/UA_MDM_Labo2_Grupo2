# Diccionario de Características Generadas (Engineered Features)

Este documento detalla todas las características (features) creadas artificialmente a partir de los datos originales en la función `feature_engineering()` del script `multimodal_ensemble.py`. 

Estas variables están diseñadas con sentido de negocio inmobiliario para capturar relaciones no lineales y exponer información oculta a los modelos de machine learning tabulares (LightGBM, XGBoost, Ridge).

---

## 1. Relaciones de Tamaño y Calidad de la Propiedad

| Feature | Fórmula / Descripción | Lógica de Negocio |
| :--- | :--- | :--- |
| `total_rooms` | `bedrooms + bathrooms` | Suma de las habitaciones principales y baños para estimar el tamaño habitable "útil" de la propiedad. |
| `area_per_room` | `livingArea / (total_rooms + 1)` | Espacio promedio por habitación. Diferencia propiedades de concepto abierto/grandes frente a distribuciones densas o cerradas. |
| `density_ratio` | `livingArea / lotAreaValue` | Relación entre la construcción y el terreno total. Propiedades urbanas suelen tener altos ratios; rurales/suburbanas tienen bajos ratios. |
| `tax_per_sqft` | `taxAssessedValue / livingArea` | Impuestos tasados por pie cuadrado. Es un indicador directo del lujo de los materiales y la exclusividad del vecindario. |
| `listing_vs_tax` | `last_listing_price / taxAssessedValue` | Mide la divergencia entre el último precio de lista y el valor fiscal oficial. Detecta propiedades sobrevaloradas o tasaciones obsoletas. |
| `hoa_per_sqft` | `hoa_fee_monthly / livingArea` | Costo de la asociación de propietarios por área. Diferencia edificios con amenidades costosas (gym, piscina) de los económicos. |
| `photos_per_room` | `photoCount / (total_rooms + 1)` | Esfuerzo fotográfico relativo al tamaño de la casa. Altos valores indican propiedades donde el vendedor quiere mostrar muchos detalles y acabados (suelen ser costosas). |

---

## 2. Antigüedad y Ciclo de Vida

| Feature | Fórmula / Descripción | Lógica de Negocio |
| :--- | :--- | :--- |
| `age` | `2024 - yearBuilt` | Antigüedad exacta de la casa en años usando el año de corte 2024. |
| `age_x_area` | `age * livingArea` | Término de interacción. Captura el hecho de que casas muy grandes *y* muy viejas pueden requerir altos costos de mantenimiento, depreciando el valor. |
| `property_age_bucket` | Segmentación de `age` en categorías. | Transforma la edad lineal en ciclos de vida inmobiliario: (0-5 años) nueva, (6-15) asentada, (16-30) requiere remodelación media, (31-50) antigua, (>50) histórica o para derribar. |

---

## 3. Percepción y Atractivo Comercial (Marketing)

| Feature | Fórmula / Descripción | Lógica de Negocio |
| :--- | :--- | :--- |
| `desc_quality_score` | Puntaje heurístico de la descripción del anuncio. | Suma palabras de la descripción (`desc_word_count`), añade bonos por menciones de *renovado* (+5), *vistas* (+3), y *piscina* (+2), y castiga fuertemente (-10) si la descripción es un texto genérico (*boilerplate*). Estima la calidad del listing. |
| `premium_flag` | 1 si (pool + waterfront + garage) >= 2. | Clasificador binario que detecta casas que concentran múltiples atributos de lujo/premium al mismo tiempo. |

---

## 4. Interacciones con Amenidades e Impuestos

| Feature | Fórmula / Descripción | Lógica de Negocio |
| :--- | :--- | :--- |
| `school_x_area` | `avg_school_rating * livingArea` | Efecto multiplicador: una casa grande en un excelente distrito escolar vale muchísimo más que una casa grande en un mal distrito. |
| `waterfront_x_area`| `has_waterfront * livingArea` | Las propiedades frente al agua se valoran exponencialmente con cada pie cuadrado extra en comparación a propiedades internas. |
| `pool_x_area` | `has_pool * livingArea` | Interacción de área con existencia de piscina. |
| `tax_rate_x_value` | `propertyTaxRate * taxAssessedValue` | Calcula el costo monetario real en impuestos que debe pagar anualmente el dueño de la propiedad. |
| `assessed_ratio` | `taxAssessedValue / last_listing_price` | A diferencia de `listing_vs_tax`, este ratio se centra en qué porcentaje del precio de mercado está cubierto por el valor tasado. |

---

## 5. Codificación Especializada (Target Encoding & Clustering)

> **Nota Técnica sobre Leakage:** Para evitar el *data leakage* (filtrado de información del futuro hacia el pasado), las codificaciones de *target* se realizan con una validación cruzada K-Fold en los datos de entrenamiento. En `test`, simplemente se aplica el valor histórico mapeado del `train`.

| Feature | Fórmula / Descripción | Lógica de Negocio |
| :--- | :--- | :--- |
| `zip_median_price_enc`| Target encoding (Mediana de `log_price` por `zipcode`) | Captura el "precio base" característico de un código postal específico utilizando la métrica menos sensible a los outliers (mediana). |
| `zip_mean_price_enc` | Target encoding (Promedio de `log_price` por `zipcode`) | Captura el valor base promediado del código postal. |
| `zip3d_median_enc` | Target encoding (Mediana por primeros 3 dígitos del zipcode) | Brinda una estimación del valor de la macrorregión postal. Muy útil para propiedades ubicadas en códigos postales poco frecuentes en el dataset de entrenamiento. |
| `geo_cluster` | Etiqueta de clustering espacial (`KMeans` con k=30) sobre `latitude` y `longitude` | Agrupa propiedades en 30 "micro-vecindarios" geográficos descubiertos automáticamente por el algoritmo, independientemente de los bordes artificiales de los códigos postales. |
| `area_rank_in_zip` | Percentil de `livingArea` dentro de su `zipcode` (0.0 a 1.0) | Mide si la casa es "pequeña" (ej. 0.1) o "mansión" (ej. 0.99) en **relación exclusiva con sus vecinos inmediatos**. Una casa grande en un barrio pobre vale diferente a una casa grande en un barrio rico. |
