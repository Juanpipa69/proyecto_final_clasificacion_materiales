# Seguridad, privacidad y gobernanza

Este documento separa lo **implementado y verificado** en el proyecto de lo que queda como **recomendación pendiente de validación**. No constituye una evaluación de cumplimiento normativo: para afirmar cumplimiento de la Ley 1581 de 2012 o del GDPR se requiere una revisión formal con el área jurídica y el oficial de protección de datos de la organización.

La evidencia cuantitativa se genera con `python -m src.evaluation.privacy_checks` y queda en `reports/privacidad_seguridad.json`.

## 1. Inventario de datos

| Fuente | Contenido | ¿Lo usa el modelo? | Datos de personas |
|---|---|---|---|
| `Maestro_materiales` | Descripción, marca, UM, precio y familia de 2.500 materiales | **Sí** | No |
| `Verdad_items_texto_libre` | 249 ítems de Ariba en texto libre | Solo como prueba externa | No |
| `Maestro_proveedores` | Nombre, país y ciudad de 400 proveedores | No | No (personas jurídicas) |
| `Ariba_eventos_RFP` | Eventos de compra, comprador, ofertas | No | Comprador **seudonimizado** (`COMP14`) |
| `ME2M_historico_compras` | Pedidos, proveedor, precios | No | Ver nota sobre `Solicitud de pedido` |

**Todos los datos son ficticios** (hoja `LEEME`: "Datos totalmente ficticios generados con semilla 42").

**Nota:** en ME2M, la columna `Solicitud de pedido` contiene números de 10 dígitos con formato similar a una cédula colombiana. En los datos simulados es un número de documento, pero con datos reales debe verificarse antes de usar esa fuente.

## 2. Medidas implementadas y verificadas

### 2.1 Protección de datos personales
| Medida | Cómo se implementa | Evidencia |
|---|---|---|
| Minimización | El modelo usa solo 7 columnas del maestro de materiales; no usa compradores, proveedores ni pedidos | `FEATURE_COLS` en `src/models/train.py` |
| Sin datos personales en las entradas | Búsqueda de correos, móviles y números de 8-10 dígitos en los textos que usa el modelo | 0 coincidencias (`privacidad_seguridad.json`) |
| Sin datos personales en el artefacto | La misma búsqueda sobre el vocabulario TF-IDF guardado dentro del modelo (4.336 términos) | 0 coincidencias |
| Logs sin contenido | La API solo registra cuántos materiales llegaron y cuánto tardó, nunca el texto | `src/api/main.py`, verificado en pruebas |
| Datos fuera de la imagen | `.dockerignore` excluye `Data/`; en Docker se montan en modo solo lectura (`:ro`) | `Dockerfile`, `docker-compose.yml` |

### 2.2 Prevención de data leakage
| Medida | Detalle |
|---|---|
| Split por grupos | 100 materiales son duplicados declarados de otros; material y duplicado quedan siempre en el mismo conjunto (`StratifiedGroupKFold` por `grupo_id`). Al corregirlo, el F1 de validación de RandomForest bajó de 1.000 a 0.988, lo que confirma que la fuga existía |
| Test intocable | El 20 % de test solo se usa una vez, al final. La selección y el tuning usan validación cruzada sobre el 80 % restante |
| Preprocesamiento dentro del Pipeline | TF-IDF, OneHot y escalado se ajustan solo con los datos de entrenamiento de cada fold |
| Etiqueta ruidosa excluida | `grupo_sap` (clasificación SAP actual) no se usa como feature |
| Prueba externa | 249 ítems con redacción distinta: F1-macro 0.992 |

### 2.3 Seguridad del modelo
| Riesgo | Medida | Resultado |
|---|---|---|
| Membership inference (saber si un material estuvo en el entrenamiento) | Ataque de caja negra por umbral de confianza | **AUC = 0.506**: el atacante no distingue (0.5 = azar). Confianza media 0.982 en entrenamiento vs. 0.979 en no vistos |
| Model inversion / extracción de datos | La API solo devuelve la familia y las 3 probabilidades principales, redondeadas a 4 decimales; nunca expone el modelo ni su vocabulario | Implementado |
| Carga insegura del modelo | MLflow guarda el modelo con **skops**, que no ejecuta código al cargar (a diferencia de pickle) y exige declarar los tipos permitidos | `src/models/tracking.py` |
| Abuso del endpoint | API key opcional (`X-API-Key`, comparación en tiempo constante), máximo 100 materiales por solicitud, validación de longitudes y rangos | Verificado: 401 sin clave, 422 con entradas inválidas |

### 2.4 Seguridad de la infraestructura
| Medida | Detalle |
|---|---|
| Usuario no privilegiado | El contenedor corre como `appuser` (uid 10001) |
| Imagen base oficial y mínima | `python:3.11-slim` |
| Dependencias fijadas | `requirements.txt` con versiones exactas |
| Secretos fuera de la imagen y del repositorio | La API key se pasa por variable de entorno al arrancar; `.env` está en `.gitignore` y `.dockerignore` |
| Exposición de puertos mínima | MLflow (5000) y API (8000) publicados solo en `127.0.0.1` |
| Trazabilidad | Cada run de MLflow registra commit de git, versiones de librerías, parámetros, métricas y artefactos |

### 2.5 Confianza y transparencia
| Medida | Detalle |
|---|---|
| Criterio de clasificación explicable | El modelo es lineal (LinearSVC) sobre palabras del texto: la decisión depende de términos identificables |
| Incertidumbre visible | Cada predicción trae su confianza y dos alternativas, para revisión humana en casos dudosos |
| Equidad entre clases | Brecha de F1 entre la mejor y la peor familia en test: 0.000 (`privacidad_seguridad.json`) |
| Comparación honesta | Líneas base (`reports/baselines.md`): la agrupación SAP actual obtiene 0.72 frente a 1.000 del modelo en test |

## 3. Limitaciones del modelo
- **Datos simulados y muy separables:** la primera palabra del texto breve basta para acertar cerca del 95 % de los casos. Con descripciones reales de SAP (abreviaturas, errores, textos ambiguos) se espera un desempeño menor.
- La prueba externa usa valores neutros para marca, UM y precio, porque esos ítems no los traen.
- El modelo no detecta materiales de familias nuevas: siempre asigna una de las 8 conocidas.
- RandomForest da resultados ligeramente distintos según la versión de las librerías; el campeón LinearSVC es determinista y dio el mismo resultado en Windows y en Docker.

## 4. Recomendaciones pendientes de validación
Estas medidas **no están implementadas**; se proponen para un uso con datos reales.

| Tema | Recomendación |
|---|---|
| Ley 1581 de 2012 / GDPR | Antes de usar datos reales: confirmar con el área jurídica la base legal del tratamiento, la finalidad y la política de tratamiento de datos de la organización. Aplicar los principios de finalidad, minimización, acceso restringido y seguridad |
| Revisión del vocabulario | Con datos reales, ejecutar `privacy_checks` en cada entrenamiento y bloquear el registro del modelo si aparecen patrones personales. Considerar `min_df >= 2` para no memorizar términos que aparecen una sola vez |
| Seudonimización | Si se incorporan Ariba o ME2M, reemplazar comprador y solicitante por identificadores con hash y sal, guardados fuera del repositorio |
| Control de acceso a MLflow | En un servidor compartido: autenticación de MLflow, HTTPS y permisos por rol; hoy solo escucha en `127.0.0.1` |
| Límite de solicitudes | Agregar rate limiting por clave o por IP (p. ej. en un proxy inverso) para dificultar ataques de extracción del modelo |
| Gestión de secretos | Usar un gestor de secretos (Azure Key Vault, AWS Secrets Manager) en lugar de variables escritas en la terminal |
| Escaneo de dependencias | Ejecutar `pip-audit` o Trivy sobre la imagen en cada versión |
| Monitoreo | Implementar `Monitoring/drift_monitoring.py` (hoy vacío) para vigilar la distribución de textos y la confianza media en producción, y reentrenar cuando cambien |
| Revisión humana | Enviar a revisión manual las predicciones con confianza menor a un umbral acordado con el negocio |
