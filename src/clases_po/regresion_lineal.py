"""Flujo reutilizable para entrenar y evaluar una regresión lineal."""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class RegresionLineal:
    """Encapsula el flujo de regresión lineal para cualquier DataFrame compatible."""

    def __init__(
        self,
        feature_column: str | list[str] | tuple[str, ...] = "Mileage",
        target_column: str = "Price",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.feature_column = self._normalizar_nombres_X(feature_column)
        if not isinstance(target_column, str):
            raise TypeError("target_column debe ser una cadena.")
        if not target_column.strip():
            raise ValueError("Los nombres de las columnas no pueden estar vacíos.")
        if target_column in self.feature_column:
            raise ValueError("La característica y el objetivo deben ser diferentes.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1.")

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.modelo: LinearRegression | None = None
        self.entrenamiento: dict[str, Any] | None = None
        self.ruta_modelo: Path | None = None
        self.predicciones: np.ndarray | None = None
        self.parametros: dict[str, float | list[float]] | None = None
        self.metricas: dict[str, float] | None = None
        self.paso_actual: str | None = None
        self.registro_reentrenamiento: pd.DataFrame | None = None

    def preparar_datos(
        self,
        df: pd.DataFrame,
        X_columnas: str | list[str] | tuple[str, ...],
        y_objetivo: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Valida un DataFrame y devuelve X e y numéricos.

        Reglas de entrada:

        - ``df`` debe ser un ``pandas.DataFrame``.
        - ``X_columnas`` puede ser un nombre ``str`` o varios nombres en una
          lista/tupla. Siempre se devuelve ``X`` como DataFrame.
          ``y_objetivo`` identifica la columna que se quiere predecir.
        - Ambas columnas deben existir y sus nombres no pueden estar vacíos.
        - Los valores se convierten a números con ``pd.to_numeric``.
          Cadenas numéricas como ``"12000"`` se convierten correctamente.
        - Las cadenas que no representan números, como ``"desconocido"``,
          se convierten temporalmente en ``NaN`` y la fila se elimina.
        - Las filas con valores nulos, infinitos o no convertibles se eliminan
          para que scikit-learn no reciba datos inválidos. La función informa
          por consola cuántas filas se eliminaron.
        - Si después de limpiar quedan menos de dos filas, se lanza
          ``ValueError`` porque no se puede entrenar una regresión.
        - Si la característica tiene un único valor, se lanza ``ValueError``:
          una recta no puede aprender una pendiente con una sola variación.

        Args:
            df: DataFrame fuente.
            X_columnas: Nombre o nombres de columnas predictoras, por ejemplo
                ``"Mileage"`` o ``["Mileage", "Year"]``.
            y_objetivo: Nombre de la columna objetivo, por ejemplo
                ``"Price"``.

        Returns:
            Una tupla ``(X, y)`` donde ``X`` es un DataFrame de una columna y
            ``y`` es una Series numérica.

        Raises:
            TypeError: Si ``df`` no es un DataFrame o los nombres no son
                cadenas.
            ValueError: Si los nombres están vacíos, no quedan filas válidas
                o la característica no tiene variación.
            KeyError: Si falta alguna columna solicitada.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        X_nombres = self._normalizar_nombres_X(X_columnas)
        if not isinstance(y_objetivo, str):
            raise TypeError("y_objetivo debe ser una cadena.")
        if not y_objetivo.strip():
            raise ValueError("Los nombres de columnas no pueden estar vacíos.")
        if y_objetivo in X_nombres:
            raise ValueError("X_columnas y y_objetivo deben ser diferentes.")

        columnas = set(X_nombres + [y_objetivo])
        faltantes = columnas.difference(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")

        datos = df[X_nombres + [y_objetivo]].copy()
        filas_iniciales = len(datos)
        datos = datos.apply(pd.to_numeric, errors="coerce")
        datos = datos.where(np.isfinite(datos), np.nan).dropna()
        filas_eliminadas = filas_iniciales - len(datos)
        self.feature_column = X_nombres
        self.target_column = y_objetivo

        if len(datos) < 2:
            raise ValueError("Se necesitan al menos dos filas numéricas válidas.")
        constantes = [nombre for nombre in X_nombres if datos[nombre].nunique() < 2]
        if constantes:
            raise ValueError(
                f"Estas características deben tener al menos dos valores "
                f"diferentes: {constantes}."
            )

        X = datos[X_nombres]
        y = datos[y_objetivo]
        self.df = df
        self.X = X
        self.y = y
        print(
            f"[preparar_datos] X={X_nombres}, y='{y_objetivo}'; "
            f"{len(datos)} filas válidas; {filas_eliminadas} filas eliminadas; "
            f"X={X.shape}, y={y.shape}."
        )
        return X, y

    def preparar_dato_uno(
        self,
        df: pd.DataFrame,
        X_columnas: str | list[str] | tuple[str, ...],
        y_objetivo: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Selecciona dos columnas que ya están limpias y son numéricas.

        Esta versión no convierte tipos, no elimina nulos y no corrige
        valores inválidos. Se utiliza únicamente cuando el llamador ya
        garantiza que ``X_columnas`` y ``y_objetivo`` existen, tienen datos
        numéricos y no contienen valores faltantes.

        Args:
            df: DataFrame previamente validado y limpio.
            X_columnas: Nombre o nombres de las columnas predictoras.
            y_objetivo: Nombre de la columna objetivo.

        Returns:
            Una tupla ``(X, y)`` con la característica como DataFrame y el
            objetivo como Series.

        Raises:
            TypeError: Si ``df`` no es un DataFrame.
            KeyError: Si alguna columna no existe.
            ValueError: Si los nombres de columnas son iguales o están vacíos.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        X_nombres = self._normalizar_nombres_X(X_columnas)
        if not isinstance(y_objetivo, str):
            raise TypeError("y_objetivo debe ser una cadena.")
        if not y_objetivo.strip():
            raise ValueError("Los nombres de columnas no pueden estar vacíos.")
        if y_objetivo in X_nombres:
            raise ValueError("X_columnas y y_objetivo deben ser diferentes.")

        faltantes = set(X_nombres + [y_objetivo]).difference(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")

        X = df[X_nombres]
        y = df[y_objetivo]
        print(
            f"[preparar_dato_uno] X={X_nombres} {X.shape}; "
            f"y='{y_objetivo}' {y.shape}; datos asumidos limpios."
        )
        return X, y

    @staticmethod
    def _normalizar_nombres_X(
        X_columnas: str | list[str] | tuple[str, ...],
    ) -> list[str]:
        """Convierte uno o varios nombres de columnas en una lista validada."""
        if isinstance(X_columnas, str):
            nombres = [X_columnas]
        elif isinstance(X_columnas, (list, tuple)):
            nombres = list(X_columnas)
        else:
            raise TypeError(
                "X_columnas debe ser una cadena, una lista o una tupla de cadenas."
            )
        if not nombres:
            raise ValueError("X_columnas debe contener al menos un nombre.")
        if not all(isinstance(nombre, str) for nombre in nombres):
            raise TypeError("Todos los nombres de X_columnas deben ser cadenas.")
        if any(not nombre.strip() for nombre in nombres):
            raise ValueError("Los nombres de X_columnas no pueden estar vacíos.")
        if len(set(nombres)) != len(nombres):
            raise ValueError("X_columnas no puede contener nombres repetidos.")
        return nombres

    @staticmethod
    def dividir_datos(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide X e y en conjuntos de entrenamiento y prueba.

        Este método es independiente del DataFrame original y de cualquier
        nombre de columna: trabaja únicamente con los pares ``X`` e ``y``
        que recibe como argumentos. Puede llamarse con una instancia o
        directamente desde la clase.

        ``test_size`` indica la proporción reservada para evaluar el modelo.
        Debe ser un número mayor que ``0`` y menor que ``1``. Por ejemplo,
        ``0.2`` reserva el 20 % para prueba y deja el 80 % para entrenamiento.
        No se acepta ``0`` porque no habría datos de prueba, ni ``1`` porque
        no habría datos para entrenar.

        ``random_state`` es una semilla entera que hace reproducible la
        separación. Usar el mismo valor produce, normalmente, la misma
        partición. Debe ser un entero mayor o igual que ``0``.

        La salida siempre es una tupla con este orden:

        ``(X_train, X_test, y_train, y_test)``

        - ``X_train``: características usadas para entrenar.
        - ``X_test``: características reservadas para evaluar.
        - ``y_train``: objetivos correspondientes a ``X_train``.
        - ``y_test``: valores reales correspondientes a ``X_test``.

        Todas las filas mantienen su correspondencia: la fila ``i`` de
        ``X_train`` corresponde al valor ``i`` de ``y_train``. La función
        imprime las cantidades y formas resultantes para dejar documentada
        la salida.

        Args:
            X: DataFrame de características.
            y: Series de valores objetivo.
            test_size: Proporción de filas para prueba, en el intervalo
                abierto ``(0, 1)``. Su valor predeterminado es ``0.2``.
            random_state: Semilla entera no negativa para reproducibilidad.
                Su valor predeterminado es ``42``.

        Returns:
            ``(X_train, X_test, y_train, y_test)``.

        Raises:
            TypeError: Si ``test_size`` no es numérico o ``random_state`` no
                es un entero.
            ValueError: Si ``test_size`` no está en ``(0, 1)``, si
                ``random_state`` es negativo o si no quedan al menos dos filas
                para entrenamiento y dos para prueba.
        """
        if not isinstance(test_size, (int, float)) or isinstance(test_size, bool):
            raise TypeError("test_size debe ser un número.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1, sin incluirlos.")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TypeError("random_state debe ser un entero.")
        if random_state < 0:
            raise ValueError("random_state debe ser mayor o igual que cero.")

        RegresionLineal._validar_pares(X, y)
        if len(X) < 2:
            raise ValueError("Se necesitan al menos dos filas para dividir los datos.")
        filas_prueba = int(np.ceil(len(X) * test_size))
        if filas_prueba < 2 or len(X) - filas_prueba < 2:
            raise ValueError(
                "Se necesitan al menos dos filas para entrenamiento y dos para prueba."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        print(
            f"[dividir_datos] test_size={test_size:.2f}; "
            f"random_state={random_state}; "
            f"entrenamiento={len(X_train)} filas {X_train.shape}; "
            f"prueba={len(X_test)} filas {X_test.shape}."
        )
        print(
            "[dividir_datos] salida: X_train/y_train entrenan el modelo; "
            "X_test/y_test miden su rendimiento con datos no vistos."
        )
        return X_train, X_test, y_train, y_test

    def entrenar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        directorio_entrenamiento: str | Path = "entrenamiento",
    ) -> dict[str, Any]:
        """Ajusta, guarda y documenta una regresión lineal.

        El modelo se guarda como un archivo ``.pkl`` dentro de
        ``directorio_entrenamiento``. El nombre incluye fecha y hora para no
        sobrescribir modelos anteriores:

        ``AAAAMMDD_HHMMSS_microsegundos_modelo.pkl``

        La función devuelve un diccionario con el modelo entrenado, los datos
        usados durante el entrenamiento y la ruta del archivo guardado.

        Args:
            X_train: DataFrame de características de entrenamiento.
            y_train: Series de objetivos de entrenamiento.
            directorio_entrenamiento: Carpeta donde se guardará el modelo.
                Por defecto es ``"entrenamiento"``.

        Returns:
            Diccionario con las claves ``modelo``, ``X_train``, ``y_train``,
            ``ruta_modelo`` y ``fecha_modelo``.

        Raises:
            TypeError: Si ``directorio_entrenamiento`` no es una ruta válida.
            ValueError: Si los datos de entrenamiento están vacíos, tienen
                longitudes incompatibles o contienen valores inválidos.
            OSError: Si no se puede crear la carpeta o escribir el archivo.
        """
        self._validar_pares(X_train, y_train)
        if not isinstance(directorio_entrenamiento, (str, Path)):
            raise TypeError("directorio_entrenamiento debe ser una ruta.")

        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        self.modelo = modelo

        carpeta = Path(directorio_entrenamiento)
        carpeta.mkdir(parents=True, exist_ok=True)
        fecha_modelo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta_modelo = carpeta / f"{fecha_modelo}_modelo.pkl"
        with ruta_modelo.open("wb") as archivo:
            pickle.dump(modelo, archivo)

        entrenamiento = {
            "modelo": modelo,
            "X_train": X_train,
            "y_train": y_train,
            "ruta_modelo": ruta_modelo,
            "fecha_modelo": fecha_modelo,
        }
        self.entrenamiento = entrenamiento
        self.ruta_modelo = ruta_modelo
        print(
            f"[entrenar] modelo ajustado con {len(X_train)} filas y "
            f"{X_train.shape[1]} características."
        )
        print(f"[entrenar] modelo guardado en: {ruta_modelo}")
        return entrenamiento

    def predecir(
        self,
        ruta_modelo: str | Path,
        X_test: float | int | list[float] | tuple[float, ...] | pd.DataFrame,
    ) -> np.ndarray:
        """Carga un ``.pkl`` y genera predicciones para uno o varios valores.

        ``X_test`` puede ser:

        - Un número, si el modelo tiene una sola característica.
        - Una lista o tupla de números, interpretada como varias filas de una
          sola característica.
        - Una lista de listas, si el modelo tiene varias características.
        - Un DataFrame con el mismo número de características del modelo.

        Args:
            ruta_modelo: Ruta del archivo ``.pkl`` creado por ``entrenar``.
            X_test: Valor individual, lista de valores o DataFrame de prueba.

        Returns:
            Un arreglo NumPy con una predicción por fila recibida.

        Raises:
            FileNotFoundError: Si no existe la ruta del modelo.
            TypeError: Si la ruta o los datos tienen un tipo incompatible.
            ValueError: Si el archivo no contiene una regresión lineal, si la
                cantidad de características no coincide o si hay valores
                inválidos.
        """
        if not isinstance(ruta_modelo, (str, Path)):
            raise TypeError("ruta_modelo debe ser una ruta.")
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")

        with ruta.open("rb") as archivo:
            modelo = pickle.load(archivo)
        if not isinstance(modelo, LinearRegression):
            raise TypeError("El archivo no contiene un modelo LinearRegression.")

        valores = self._normalizar_X_test(X_test, modelo.n_features_in_)
        if valores.empty:
            raise ValueError("X_test no puede estar vacío.")
        if hasattr(modelo, "feature_names_in_"):
            valores.columns = modelo.feature_names_in_

        predicciones = modelo.predict(valores)
        self.predicciones = predicciones
        print(
            f"[predecir] modelo cargado desde {ruta}; "
            f"{len(predicciones)} predicciones generadas."
        )
        return predicciones

    @staticmethod
    def _normalizar_X_test(
        X_test: float | int | list[float] | tuple[float, ...] | pd.DataFrame,
        numero_caracteristicas: int,
    ) -> pd.DataFrame:
        """Convierte valores de prueba a un DataFrame con forma compatible."""
        if isinstance(X_test, pd.DataFrame):
            valores = X_test.copy()
        elif isinstance(X_test, (int, float)) and not isinstance(X_test, bool):
            if numero_caracteristicas != 1:
                raise ValueError(
                    "Un valor individual solo sirve para un modelo "
                    "con una característica."
                )
            valores = pd.DataFrame([[X_test]])
        elif isinstance(X_test, (list, tuple)):
            if not X_test:
                raise ValueError("La lista X_test no puede estar vacía.")
            arreglo = np.asarray(X_test, dtype=float)
            if arreglo.ndim == 1:
                if numero_caracteristicas == 1:
                    arreglo = arreglo.reshape(-1, 1)
                elif len(arreglo) == numero_caracteristicas:
                    arreglo = arreglo.reshape(1, -1)
                else:
                    raise ValueError(
                        "Una lista plana no coincide con las características "
                        "del modelo."
                    )
            valores = pd.DataFrame(arreglo)
        else:
            raise TypeError(
                "X_test debe ser un número, una lista, una tupla o un DataFrame."
            )

        if valores.shape[1] != numero_caracteristicas:
            raise ValueError(
                f"El modelo espera {numero_caracteristicas} características y "
                f"X_test contiene {valores.shape[1]}."
            )
        valores = valores.apply(pd.to_numeric, errors="coerce")
        if valores.isna().any().any() or not np.isfinite(valores.to_numpy()).all():
            raise ValueError("X_test contiene valores no numéricos o inválidos.")
        return valores

    def obtener_parametros(
        self, ruta_modelo: str | Path
    ) -> dict[str, float | list[float]]:
        """Carga un ``.pkl`` y devuelve ``w`` y ``b`` del modelo guardado.

        ``w`` es un número cuando el modelo tiene una característica y una
        lista cuando tiene varias. El orden de la lista coincide con el orden
        de las columnas utilizadas para entrenar el modelo.

        Args:
            ruta_modelo: Ruta del archivo ``.pkl`` creado por ``entrenar``.

        Returns:
            Diccionario con ``w`` (pendiente o coeficientes) y ``b``
            (intercepto).

        Raises:
            FileNotFoundError: Si no existe la ruta del modelo.
            TypeError: Si la ruta no es válida o el archivo no contiene una
                regresión lineal.
        """
        if not isinstance(ruta_modelo, (str, Path)):
            raise TypeError("ruta_modelo debe ser una ruta.")
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")

        with ruta.open("rb") as archivo:
            modelo = pickle.load(archivo)
        if not isinstance(modelo, LinearRegression):
            raise TypeError("El archivo no contiene un modelo LinearRegression.")

        coeficientes = modelo.coef_.astype(float).tolist()
        w: float | list[float] = (
            coeficientes[0] if len(coeficientes) == 1 else coeficientes
        )
        b = float(modelo.intercept_)
        parametros = {"w": w, "b": b}
        self.parametros = parametros
        print(
            f"[obtener_parametros] modelo cargado desde {ruta}; "
            f"w={w}; b={b:.4f}."
        )
        return parametros

    def evaluar(
        self,
        ruta_modelo: str | Path,
        X_test: float | int | list[float] | tuple[float, ...] | pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Carga un modelo, predice y calcula sus métricas de prueba.

        La evaluación utiliza siempre el modelo guardado en ``ruta_modelo`` y
        genera las predicciones a partir de ``X_test``. Así no depende de un
        objeto modelo que permanezca en memoria ni de un DataFrame concreto.

        Args:
            ruta_modelo: Ruta del archivo ``.pkl`` creado por ``entrenar``.
            X_test: Un valor, una lista de valores o un DataFrame de prueba,
                compatible con ``predecir``.
            y_test: Valores reales correspondientes a ``X_test``.

        Returns:
            Diccionario con ``mae``, ``rmse`` y ``r2``.

        Raises:
            FileNotFoundError: Si no existe el modelo.
            TypeError: Si ``y_test`` no es una Series.
            ValueError: Si los valores reales o las predicciones no son
                vectores finitos de la misma longitud.
        """
        if not isinstance(y_test, pd.Series):
            raise TypeError("y_test debe ser una pandas.Series.")

        predicciones = self.predecir(ruta_modelo, X_test)
        reales = np.asarray(y_test, dtype=float)
        estimadas = np.asarray(predicciones, dtype=float)
        if reales.ndim != 1 or estimadas.ndim != 1:
            raise ValueError("y_test y predicciones deben ser vectores de una dimensión.")
        if len(reales) == 0 or len(reales) != len(estimadas):
            raise ValueError("y_test y predicciones deben tener la misma longitud.")
        if not np.isfinite(reales).all() or not np.isfinite(estimadas).all():
            raise ValueError("y_test y predicciones no pueden contener valores inválidos.")

        resultados = {
            "mae": float(mean_absolute_error(reales, estimadas)),
            "rmse": float(np.sqrt(mean_squared_error(reales, estimadas))),
            "r2": float(r2_score(reales, estimadas)),
        }
        self.metricas = resultados
        print(
            f"[evaluar] modelo={ruta_modelo}; "
            f"MAE={resultados['mae']:.2f}; "
            f"RMSE={resultados['rmse']:.2f}; R2={resultados['r2']:.2f}."
        )
        return resultados

    def reentrenar(
        self,
        df: pd.DataFrame,
        ruta_modelo_actual: str | Path | None = None,
        ruta_registro: str | Path = "entrenamiento/registro_modelos.csv",
        metrica: str = "r2",
        mayor_es_mejor: bool = True,
    ) -> dict[str, Any]:
        """Registra el modelo actual o compara un nuevo reentrenamiento.

        La primera ejecución evalúa el modelo actual y crea el CSV de
        seguimiento. Las siguientes ejecuciones entrenan un modelo nuevo,
        comparan su métrica con la última registrada y añaden una fila al CSV.

        ``mayor_es_mejor=True`` es apropiado para ``r2``. Para ``mae`` o
        ``rmse`` debe usarse ``mayor_es_mejor=False``, porque en esas métricas
        un valor menor representa un modelo mejor.

        La columna ``comparacion`` indica ``inicial``, ``igual``, ``subio`` o
        ``bajo``. ``diferencia`` usa la dirección de mejora: es positiva si
        el nuevo modelo mejora y negativa si empeora. La alerta se calcula
        con la degradación porcentual:

        - ``sin_alerta``: no baja o baja menos del 5 %.
        - ``baja``: baja desde 5 % hasta menos del 10 %.
        - ``media``: baja desde 10 % hasta menos del 20 %.
        - ``alta``: baja desde 20 % hasta menos del 35 %.
        - ``critica``: baja 35 % o más.

        El CSV contiene la fecha extraída del nombre del ``.pkl``, la ruta,
        la métrica, el resultado de la comparación, la alerta y las
        diferencias para poder auditar cada versión.

        Args:
            df: DataFrame que se utilizará para preparar y evaluar los datos.
            ruta_modelo_actual: Ruta del modelo actual. Si es ``None``, usa
                ``self.ruta_modelo``.
            ruta_registro: Ruta del CSV de seguimiento.
            metrica: ``"r2"``, ``"mae"`` o ``"rmse"``.
            mayor_es_mejor: Indica la dirección de mejora de la métrica.

        Returns:
            Diccionario con el registro actualizado, la ruta del CSV y, si
            hubo reentrenamiento, la información del nuevo modelo.

        Raises:
            FileNotFoundError: Si no existe el modelo actual.
            ValueError: Si la métrica o los datos no son válidos.
        """
        if metrica not in {"r2", "mae", "rmse"}:
            raise ValueError("metrica debe ser 'r2', 'mae' o 'rmse'.")
        if not isinstance(mayor_es_mejor, bool):
            raise TypeError("mayor_es_mejor debe ser booleano.")
        if ruta_modelo_actual is None:
            ruta_modelo_actual = self.ruta_modelo
        if ruta_modelo_actual is None:
            raise ValueError("No hay un modelo actual para reentrenar.")

        ruta_modelo_actual = Path(ruta_modelo_actual)
        if not ruta_modelo_actual.is_file():
            raise FileNotFoundError(f"No existe el modelo actual: {ruta_modelo_actual}.")

        ruta_csv = Path(ruta_registro)
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        X, y = self.preparar_datos(df, self.feature_column, self.target_column)
        X_train, X_test, y_train, y_test = self.dividir_datos(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        metricas_actuales = self.evaluar(ruta_modelo_actual, X_test, y_test)
        valor_actual = metricas_actuales[metrica]
        fecha_actual = self._fecha_modelo(ruta_modelo_actual)

        columnas = [
            "fecha_modelo",
            "ruta_modelo",
            "metrica_nombre",
            "metrica",
            "metrica_anterior",
            "comparacion",
            "diferencia",
            "diferencia_porcentaje",
            "nivel_alerta",
            "fecha_registro",
        ]
        if ruta_csv.exists():
            registro = pd.read_csv(ruta_csv)
            faltantes = set(columnas).difference(registro.columns)
            if faltantes:
                raise ValueError(
                    f"El CSV no contiene las columnas requeridas: {sorted(faltantes)}."
                )
        else:
            registro = pd.DataFrame(columns=columnas)

        if registro.empty:
            fila = self._fila_registro(
                fecha_actual,
                ruta_modelo_actual,
                metrica,
                valor_actual,
                None,
                "inicial",
                0.0,
                0.0,
                "sin_alerta",
            )
            registro = pd.DataFrame([fila], columns=columnas)
            registro.to_csv(ruta_csv, index=False)
            self.registro_reentrenamiento = registro
            print(
                f"[reentrenar] registro inicial creado para {ruta_modelo_actual}; "
                f"{metrica}={valor_actual:.4f}."
            )
            return {
                "accion": "registro_inicial",
                "ruta_modelo": ruta_modelo_actual,
                "ruta_registro": ruta_csv,
                "metrica": valor_actual,
                "registro": registro,
            }

        ultima = registro.iloc[-1]
        valor_anterior = float(ultima["metrica"])
        entrenamiento = self.entrenar(X_train, y_train)
        ruta_nueva = entrenamiento["ruta_modelo"]
        metricas_nuevas = self.evaluar(ruta_nueva, X_test, y_test)
        valor_nuevo = metricas_nuevas[metrica]
        diferencia = (
            valor_nuevo - valor_anterior
            if mayor_es_mejor
            else valor_anterior - valor_nuevo
        )
        porcentaje = (
            diferencia / abs(valor_anterior) * 100
            if valor_anterior != 0
            else (0.0 if diferencia == 0 else 100.0)
        )
        comparacion = "igual" if np.isclose(diferencia, 0.0) else (
            "subio" if diferencia > 0 else "bajo"
        )
        nivel_alerta = self._nivel_alerta(porcentaje)
        fila = self._fila_registro(
            self._fecha_modelo(ruta_nueva),
            ruta_nueva,
            metrica,
            valor_nuevo,
            valor_anterior,
            comparacion,
            diferencia,
            porcentaje,
            nivel_alerta,
        )
        registro = pd.concat([registro, pd.DataFrame([fila])], ignore_index=True)
        registro.to_csv(ruta_csv, index=False)
        self.registro_reentrenamiento = registro
        if comparacion == "bajo":
            print(
                f"[reentrenar] ALERTA {nivel_alerta}: la métrica bajó "
                f"{abs(porcentaje):.2f}%."
            )
        else:
            print(
                f"[reentrenar] nuevo modelo registrado; "
                f"comparación={comparacion}, diferencia={diferencia:.4f}."
            )
        return {
            "accion": "reentrenamiento",
            "ruta_modelo": ruta_nueva,
            "ruta_registro": ruta_csv,
            "metrica": valor_nuevo,
            "metrica_anterior": valor_anterior,
            "comparacion": comparacion,
            "diferencia": diferencia,
            "diferencia_porcentaje": porcentaje,
            "nivel_alerta": nivel_alerta,
            "registro": registro,
        }

    @staticmethod
    def _fecha_modelo(ruta_modelo: Path) -> str:
        """Extrae la fecha textual del nombre estándar del modelo."""
        return ruta_modelo.stem.removesuffix("_modelo")

    @staticmethod
    def _nivel_alerta(diferencia_porcentaje: float) -> str:
        """Clasifica la degradación porcentual del nuevo modelo."""
        degradacion = max(0.0, -diferencia_porcentaje)
        if degradacion < 5:
            return "sin_alerta"
        if degradacion < 10:
            return "baja"
        if degradacion < 20:
            return "media"
        if degradacion < 35:
            return "alta"
        return "critica"

    @staticmethod
    def _fila_registro(
        fecha_modelo: str,
        ruta_modelo: Path,
        metrica_nombre: str,
        metrica: float,
        metrica_anterior: float | None,
        comparacion: str,
        diferencia: float,
        diferencia_porcentaje: float,
        nivel_alerta: str,
    ) -> dict[str, Any]:
        """Construye una fila consistente para el CSV de seguimiento."""
        return {
            "fecha_modelo": fecha_modelo,
            "ruta_modelo": str(ruta_modelo),
            "metrica_nombre": metrica_nombre,
            "metrica": metrica,
            "metrica_anterior": metrica_anterior,
            "comparacion": comparacion,
            "diferencia": diferencia,
            "diferencia_porcentaje": diferencia_porcentaje,
            "nivel_alerta": nivel_alerta,
            "fecha_registro": datetime.now().isoformat(timespec="seconds"),
        }

    def ejecutar(self, df: pd.DataFrame) -> dict[str, Any]:
        """Ejecuta todo el flujo y conserva cada resultado en la instancia.

        El método recibe el DataFrame una sola vez y ejecuta, en orden,
        preparación, división, entrenamiento, predicción, extracción de
        parámetros y evaluación. Cada etapa informa su nombre por consola y
        deja su resultado disponible en atributos como ``self.X_train``,
        ``self.ruta_modelo`` o ``self.metricas``.
        """
        self.paso_actual = "preparar_datos"
        print("[ejecutar][1/6] Iniciando preparar_datos.")
        X, y = self.preparar_datos(
            df,
            self.feature_column,
            self.target_column,
        )
        self.paso_actual = "dividir_datos"
        print("[ejecutar][2/6] Iniciando dividir_datos.")
        X_train, X_test, y_train, y_test = self.dividir_datos(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.paso_actual = "entrenar"
        print("[ejecutar][3/6] Iniciando entrenar.")
        entrenamiento = self.entrenar(X_train, y_train)
        modelo = entrenamiento["modelo"]
        self.paso_actual = "predecir"
        print("[ejecutar][4/6] Iniciando predecir.")
        predicciones = self.predecir(entrenamiento["ruta_modelo"], X_test)
        self.paso_actual = "obtener_parametros"
        print("[ejecutar][5/6] Iniciando obtener_parametros.")
        parametros = self.obtener_parametros(entrenamiento["ruta_modelo"])
        self.paso_actual = "evaluar"
        print("[ejecutar][6/6] Iniciando evaluar.")
        metricas = self.evaluar(
            entrenamiento["ruta_modelo"],
            X_test,
            y_test,
        )
        self.paso_actual = "completado"
        print("[ejecutar] flujo completo terminado.")
        return {
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "modelo": modelo,
            "entrenamiento": entrenamiento,
            "ruta_modelo": entrenamiento["ruta_modelo"],
            "predicciones": predicciones,
            "parametros": parametros,
            "metricas": metricas,
        }

    @staticmethod
    def _validar_pares(X: pd.DataFrame, y: pd.Series) -> None:
        """Comprueba que las características y objetivos forman un par válido.

        En aprendizaje supervisado, cada fila de ``X`` debe corresponder al
        valor situado en la misma posición de ``y``. Por ejemplo, la fila
        ``X.iloc[3]`` debe describir el mismo registro que ``y.iloc[3]``.
        Esta relación es necesaria para que el modelo aprenda la asociación
        correcta entre las entradas y sus respuestas.

        La validación comprueba que:

        - ``X`` sea un ``DataFrame`` y ``y`` sea una ``Series``.
        - Ninguno de los dos objetos esté vacío.
        - Ambos tengan exactamente la misma cantidad de filas.
        - Todos los valores de ``X`` e ``y`` sean numéricos y finitos.

        Su importancia es evitar que los métodos de división, entrenamiento o
        predicción trabajen con datos desalineados, vacíos, textuales, nulos,
        infinitos o con longitudes incompatibles. Sin esta comprobación, el
        error podría aparecer más tarde dentro de scikit-learn o, peor, el
        modelo podría entrenarse con pares entrada-respuesta incorrectos.

        Args:
            X: DataFrame con una fila por observación y una o más
                características.
            y: Series con una respuesta por cada fila de ``X``.

        Raises:
            TypeError: Si ``X`` o ``y`` no tienen el tipo esperado.
            ValueError: Si están vacíos, tienen distinta longitud o contienen
                valores no numéricos, nulos o infinitos.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y debe ser una pandas.Series.")
        if X.empty or y.empty or len(X) != len(y):
            raise ValueError("X e y deben tener filas y longitudes compatibles.")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("X contiene valores no numéricos o inválidos.")
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("y contiene valores no numéricos o inválidos.")