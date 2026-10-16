# ML_202604_Santiago_Lopez_F
Repositorio de referencia: un notebook ejecutable por clase que deja un resumen .txt en reportes/.

Estructura
src/rp_utils/            Clases reutilizables
  reporting.py           ClassReport: genera el resumen .txt de cada notebook
  hf_loader.py           HFDatasetLoader: conexión automática a Hugging Face y exportación
  summarizer.py          DatasetSummarizer: resumen de un DataFrame
sincronicas/             8 notebooks: teoría + demostración por clase
triplea/                 8 notebooks: práctica con ejercicios por clase
Actividades_Evaluativas/ 4 notebooks con la solución de referencia
reportes/                Resúmenes .txt generados (uno por notebook)
scripts/build_notebooks.py  Regenera todos los notebooks
Temario (propuesto, ajustable en scripts/build_notebooks.py)
Clase	Tema	Unidad
1	Introducción a Python y entorno de análisis	1
2	NumPy y computación vectorizada	1
3	pandas: carga y exploración	1
4	Limpieza y transformación	2
5	Visualización	2
6	Estadística descriptiva e inferencial	2
7	Ingeniería de características	3
8	Modelado básico y evaluación	3
Actividades evaluativas
Conexión automática a Hugging Face con HFDatasetLoader y resumen con DatasetSummarizer, orquestado desde el notebook.
Unidad 1: exploración con Python, NumPy y pandas.
Unidad 2: limpieza, visualización y estadística.
Unidad 3: ingeniería de características y modelado.
El dataset se cambia editando DATASET, CONFIG, SPLIT y TARGET en la primera celda de cada actividad.

Uso
cd /workspaces/RP_AD2026Q4
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow openpyxl  # opcional: parquet y xlsx
python scripts/build_notebooks.py   # opcional: regenerar notebooks
jupyter lab
En Bash, el entorno virtual debe activarse con source (o con un punto):

source .venv/bin/activate
# equivalente:
. .venv/bin/activate
No se debe ejecutar .venv/bin/activate directamente, porque activate es un script que modifica la sesión actual de la terminal. Cuando se activa correctamente, el prompt muestra (.venv).

Cada notebook termina con report.save(), que escribe reportes/<carpeta>/<notebook>.txt.

Autenticación en Hugging Face
Para descargar datasets públicos normalmente no se necesita un token. Si el dataset es privado o requiere aceptación de condiciones de uso, configura un token de Hugging Face siguiendo estos pasos.

1. Crear un token
Inicia sesión en Hugging Face.
Abre Settings > Access Tokens.
Selecciona Create new token.
Asigna un nombre descriptivo, por ejemplo RP_AD2026Q4.
Selecciona el permiso mínimo necesario: Read.
Presiona Create token y copia el token. Normalmente comienza por hf_.
Guarda el token de forma segura y no lo incluyas en el código fuente.

2. Instalar la CLI e iniciar sesión
Con el entorno virtual activado, instala o actualiza la librería de Hugging Face:

python -m pip install --upgrade huggingface_hub
Inicia sesión desde la terminal:

hf auth login
Cuando la terminal solicite el token, pégalo y presiona Enter. Es normal que no se muestren caracteres mientras lo escribes o pegas.

La sesión queda guardada localmente para que HFDatasetLoader pueda usarla automáticamente. Comprueba que la autenticación funcionó:

hf auth whoami
Si tu instalación usa la CLI antigua, el comando equivalente es:

huggingface-cli login
3. Cargar el dataset autenticado
Después de iniciar sesión, no es necesario pasar el token al constructor:

from rp_utils import HFDatasetLoader

loader = HFDatasetLoader(
    "Fast_Food_Restaurants",
    filename="datos/fast_food.csv",
)
df = loader.load()
No escribas tokens directamente en archivos Python ni los subas a GitHub. Si un token se expone accidentalmente, revócalo desde Access Tokens y crea uno nuevo.

Carga de un dataset como DataFrame
HFDatasetLoader recibe el identificador del dataset, descarga la división indicada y devuelve un pandas.DataFrame. filename y file_type son opcionales; si se proporciona filename, el DataFrame también se guarda en csv, json, parquet o xlsx.

from rp_utils import HFDatasetLoader

loader = HFDatasetLoader(
    "Fast_Food_Restaurants",
    filename="datos/fast_food.csv",
)
df = loader.load()
Para ejecutar este ejemplo manualmente, crea un archivo prueba_loader.py en la raíz del proyecto con el código anterior y ejecuta:

PYTHONPATH=src python prueba_loader.py
También se puede ejecutar directamente desde la terminal:

PYTHONPATH=src python -c '
from rp_utils import HFDatasetLoader

df = HFDatasetLoader(
    "Fast_Food_Restaurants",
    filename="datos/fast_food.csv",
).load()

print(df.head())
'
También se puede indicar el formato explícitamente:

loader = HFDatasetLoader(
    "Fast_Food_Restaurants",
    filename="datos/fast_food",
    file_type="parquet",
)
df = loader.load()