# ML_202604_Santiago_Lopez_F
# Documento de Hallazgos: Segmentación del Mercado de Arriendos

Documento de Hallazgos: Segmentación del Mercado de Arriendos (Modelo O.V.C.)
# Respuesta a la Pregunta de Negocio
¿Qué segmentos de barrios existen en la ciudad según su nivel de precios y oferta para diseñar campañas de marketing personalizadas?

El análisis determinó que el mercado inmobiliario se divide en 4 segmentos diferenciados, demostrando que una estrategia de marketing generalista es ineficiente. La dinámica de oferta y valor está altamente concentrada; un pequeño grupo de zonas domina la mayoría del volumen transaccional, requiriendo que la asignación de presupuesto publicitario y los esfuerzos comerciales se focalicen según el perfil del barrio.

# Hallazgos Principales (Sustentados por K-Means, k=4)
El modelo agrupó los barrios considerando la Oferta (volumen de avisos) y el Valor (mediana del precio de arriendo), logrando un coeficiente de Silueta de 0.357.

-> Zonas Premium (Alto Costo y Oferta): El motor del negocio.

* Este segmento concentra el mayor volumen del mercado, destacándose por tener niveles de oferta muy por encima del promedio de la ciudad (como se observa en el pico superior del gráfico de centroides z).   
* En el gráfico de dispersión (puntos verdes), estos barrios se extienden horizontalmente hacia la derecha, evidenciando volúmenes de oferta que superan ampliamente los 1,000 avisos, manteniendo precios medios-altos.   

->Zonas Dinámicas (Mercado Activo): El nicho de lujo.

* Este grupo representa las propiedades más exclusivas. Su huella en el gráfico de centroides (línea azul) muestra que el "Valor" está muy por encima del promedio de la ciudad.
* Visualmente (puntos azules), se ubican en la parte superior izquierda del plano: tienen precios altísimos (mediana en el rango de los 10.000.000 COP), pero con un volumen de oferta bajo (generalmente menos de 100 avisos por zona).   

-> Zonas Económicas o Populares (1 y 3): La base territorial de bajo impacto comercial.

* Estos dos grupos conforman la mayoría de los barrios (puntos rojos y naranjas Grafica de segmentos).   
* Se caracterizan por tener baja oferta (agrupados a la izquierda del gráfico, usualmente por debajo de los 100 avisos) y valores por debajo del promedio de la ciudad (líneas roja y naranja en el gráfico de centroides).   
* Nota sobre infraestructura: Un análisis preliminar de la variable extra TasaParqueadero sugiere que estas zonas tienen carencias estructurales. Por ejemplo, en el segmento 3 (barrio Abraham Lincoln), la tasa de parqueadero registrada fue de 0.000000, indicando una ausencia total de esta amenidad en los avisos evaluados.   

# Confiabilidad y Limitaciones
Confiabilidad: El modelo procesó datos completamente actualizados; todos los barrios analizados mostraron una Recencia_Datos de 1 día (el valor mínimo posible frente a la fecha de corte). Esto garantiza que la segmentación refleja la realidad actual del mercado y no un escenario pasado.   

Limitaciones: La segmentación asume al "barrio" como una unidad homogénea, promediando el precio (Valor) y la cantidad de inmuebles (Oferta). El modelo no diferencia entre micro-sectores dentro de un mismo barrio ni considera los días en el mercado (velocidad de absorción) de los inmuebles.

# Recomendaciones Estratégicas
1. Focalización de Pauta Digital (Zonas Premium): Dirigir el grueso del presupuesto de marketing (Google Ads, pauta en portales) hacia las Zonas Premium. Estas zonas tienen la liquidez (oferta masiva) y la rentabilidad necesarias para justificar la inversión en adquisición de clientes.

2. Marketing Boutique y Relacional (Zonas Dinámicas): Para los barrios de alto lujo, evitar las campañas masivas. La estrategia debe basarse en networking, agentes especializados, catálogos exclusivos y alianzas con clientes corporativos, ya que el volumen de propiedades es bajo pero la comisión por transacción es significativamente alta.

3. Eficiencia Operativa (Zonas Económicas): Dado el bajo volumen y el menor precio de estas propiedades, es vital reducir el costo de adquisición (CAC). Se recomienda automatizar la captación y atención en estos barrios mediante embudos de self-service, visitas virtuales y firmas digitales, minimizando el tiempo invertido por los asesores comerciales.

4. Uso del Atributo "Parqueadero" como Filtro Premium: Para futuras campañas, cruzar la segmentación actual con la TasaParqueadero. Los inmuebles con parqueadero en Zonas Económicas pueden mercadearse como "Oportunidades Destacadas", mientras que en las Zonas Premium, el parqueadero debe asumirse como un requisito mínimo indispensable.