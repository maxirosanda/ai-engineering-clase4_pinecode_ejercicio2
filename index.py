# =====================================================================
# EJERCICIO 2: ingesta en lotes con metadatos y búsqueda filtrada
# =====================================================================
# Continúa el Ejercicio 1: usa el MISMO índice ("mi-indice-serverless").
#
# ¿Qué hace este programa? Tres cosas, en orden:
#   1. INGESTA: toma 5 documentos de texto, los convierte en vectores
#      con OpenAI y los sube a Pinecone EN LOTES (de a 2).
#   2. BÚSQUEDA: hace una pregunta y busca la respuesta SOLO entre los
#      documentos de la categoría "legal" (filtro por metadatos).
#   3. ESTADÍSTICAS: muestra que el índice ahora tiene dos namespaces.
#
# Antes de correrlo (una sola vez), en la terminal:
#     pip install pinecone openai python-dotenv
# y en tu archivo .env (en la misma carpeta que este archivo):
#     PINECONE_API_KEY=...
#     OPENAI_API_KEY=...
# =====================================================================

# ---------------------------------------------------------------------
# IMPORTS: las herramientas que vamos a usar
# ---------------------------------------------------------------------
import os                               # para leer variables de entorno (las claves)
import time                             # para hacer una pausa con time.sleep()
from datetime import datetime           # para guardar la fecha y hora de carga
from typing import List, Dict, Any      # "pistas de tipo": documentan qué recibe y
                                        # devuelve cada función (Python no las exige)
from dotenv import load_dotenv          # lee el archivo .env
from openai import OpenAI               # cliente de OpenAI: genera los embeddings
from pinecone import Pinecone           # cliente de Pinecone: guarda y busca vectores

# Carga las variables del archivo .env como si fueran variables de entorno.
# Después de esta línea, os.getenv("PINECONE_API_KEY") ya devuelve tu clave.
load_dotenv()

# ---------------------------------------------------------------------
# CONSTANTES: valores fijos que usa todo el programa.
# Se escriben en MAYÚSCULAS por convención. Tenerlos en un solo lugar
# evita errores: si hay que cambiar algo, se cambia acá y listo.
# ---------------------------------------------------------------------
INDEX_NAME = "mi-indice-serverless"          # el índice que creó el Ejercicio 1
NAMESPACE = "documentos"                     # otra partición, separada de "dev-environment"
EMBEDDING_MODEL = "text-embedding-3-small"   # genera vectores de 1536 números:
                                             # TIENE que coincidir con la dimensión
                                             # del índice, o Pinecone da error


# ---------------------------------------------------------------------
# EL PIPELINE DE INGESTA
# ---------------------------------------------------------------------
# Una CLASE agrupa datos y funciones que trabajan juntos.
# Acá juntamos todo lo necesario para cargar y buscar documentos:
# la conexión a Pinecone, la conexión a OpenAI y las funciones que las usan.
class IngestionPipeline:

    # __init__ se ejecuta UNA vez, al crear el objeto:
    #     pipeline = IngestionPipeline(index, OpenAI())
    # Guarda las dos conexiones en "self" para que las demás funciones
    # de la clase puedan usarlas (self.index y self.openai).
    def __init__(self, index, openai_client: OpenAI):
        self.index = index              # el índice de Pinecone (donde se guarda)
        self.openai = openai_client     # el cliente de OpenAI (quien vectoriza)

    # -----------------------------------------------------------------
    # embed: texto → vectores
    # -----------------------------------------------------------------
    # Recibe una LISTA de textos y devuelve una LISTA de vectores,
    # en el mismo orden. Ejemplo:
    #     embed(["hola", "chau"])  →  [[0.01, -0.3, ...], [0.2, 0.05, ...]]
    #                                   (1536 números)    (1536 números)
    # Importante: es UNA sola llamada a OpenAI para todo el lote,
    # no una por texto. Menos llamadas = más rápido y más barato.
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)

        # response.data es una lista con un resultado por cada texto.
        # De cada resultado nos quedamos solo con .embedding (el vector).
        # Esto es una "list comprehension": una forma corta de escribir
        #     vectores = []
        #     for item in response.data:
        #         vectores.append(item.embedding)
        return [item.embedding for item in response.data]

    # -----------------------------------------------------------------
    # create_metadata: arma las "etiquetas" de cada vector
    # -----------------------------------------------------------------
    # Es la ÚNICA función del programa que arma metadatos. Así todos los
    # vectores tienen las mismas claves, con el mismo formato.
    # Esto evita el "schema drift" (ver la diapositiva del tema):
    # que un proceso guarde "ingested_at" y otro "fecha", y el filtro
    # ignore en silencio a la mitad de los documentos.
    def create_metadata(self, doc_text: str, category: str, author: str) -> Dict[str, Any]:
        return {
            # El texto original. Pinecone guarda NÚMEROS, no texto: si no lo
            # ponemos acá, al buscar recuperaríamos un vector que no podemos
            # leer. Con esto, el resultado ya trae el texto listo para el LLM.
            "text": doc_text,

            # Las etiquetas por las que después vamos a poder FILTRAR
            "category": category,           # "legal", "hr", "finanzas"
            "author": author,               # quién cargó el documento

            # Fecha y hora de carga. isoformat() da un texto estándar:
            #     "2026-11-24T08:30:00.123456"
            # Siempre el mismo formato → se puede comparar y ordenar.
            "ingested_at": datetime.now().isoformat(),

            # Tamaño del texto en caracteres: sirve, por ejemplo, para
            # detectar documentos vacíos o demasiado largos.
            "char_count": len(doc_text),
        }

    # -----------------------------------------------------------------
    # process_and_upsert_batches: carga TODOS los documentos, de a lotes
    # -----------------------------------------------------------------
    # documents:  lista de diccionarios, cada uno con "text", "category" y "author"
    # batch_size: cuántos documentos van en cada lote (100 por defecto)
    # Devuelve:   la cantidad total de vectores subidos
    def process_and_upsert_batches(self, documents: List[Dict], batch_size: int = 100) -> int:
        total = 0   # contador de vectores subidos

        # range(inicio, fin, salto) genera números saltando de a "salto".
        # Con 5 documentos y batch_size=2:  range(0, 5, 2) → 0, 2, 4
        # "i" NO es un documento: es la POSICIÓN donde empieza cada lote.
        for i in range(0, len(documents), batch_size):

            # 1. Recortar el lote: lista[desde : hasta] (sin incluir "hasta")
            #      i = 0 → documents[0:2] → documentos 0 y 1
            #      i = 2 → documents[2:4] → documentos 2 y 3
            #      i = 4 → documents[4:6] → solo el documento 4
            #    En la última vuelta "hasta" (6) se pasa del largo (5), pero
            #    Python NO da error: devuelve lo que haya. Por eso el último
            #    lote sale más chico.
            batch = documents[i : i + batch_size]

            # 2. Sacar los textos del lote y pedir todos sus embeddings juntos.
            #    [doc["text"] for doc in batch] → ["Contrato de...", "Acuerdo de..."]
            #    embeddings queda con un vector por documento, en el mismo orden.
            embeddings = self.embed([doc["text"] for doc in batch])

            # 3. Armar los vectores en el formato que espera Pinecone:
            #    un diccionario con "id", "values" y "metadata".
            #
            #    zip(batch, embeddings) junta cada documento con SU vector:
            #        (doc0, vector0), (doc1, vector1)
            #    enumerate(...) les agrega un número de orden "j" dentro del lote:
            #        j=0 → (doc0, vector0),  j=1 → (doc1, vector1)
            #
            #    El id se calcula con la posición GLOBAL del documento:
            #        i + j + 1  →  en el lote que empieza en i=2:
            #                      j=0 → "doc-3",  j=1 → "doc-4"
            #    (el +1 es para que empiece en "doc-1" y no en "doc-0")
            #
            #    El id es FIJO: si corrés el programa otra vez, "doc-1" sigue
            #    siendo "doc-1", y upsert lo ACTUALIZA en vez de duplicarlo.
            vectors = [
                {
                    "id": f"doc-{i + j + 1}",       # identificador único
                    "values": embedding,            # los 1536 números
                    "metadata": self.create_metadata(
                        doc["text"], doc["category"], doc["author"]),
                }
                for j, (doc, embedding) in enumerate(zip(batch, embeddings))
            ]

            # 4. Subir el lote completo en UNA llamada a Pinecone.
            #    upsert = update + insert: si el id existe, lo actualiza;
            #    si no existe, lo crea.
            #    namespace=NAMESPACE lo guarda en la partición "documentos",
            #    sin tocar los vectores de prueba del Ejercicio 1.
            self.index.upsert(vectors=vectors, namespace=NAMESPACE)
            print(f"Lote subido: {len(vectors)} vectores")
            total += len(vectors)           # igual a: total = total + len(vectors)

        return total

    # -----------------------------------------------------------------
    # search_by_category: busca por significado, pero solo en una categoría
    # -----------------------------------------------------------------
    # question: la pregunta del usuario, en texto
    # category: la categoría donde buscar ("legal", "hr", ...)
    # top_k:    cuántos resultados queremos como MÁXIMO
    def search_by_category(self, question: str, category: str, top_k: int = 5):

        # 1. Pasar la pregunta a vector con el MISMO modelo que los documentos.
        #    Si usáramos otro modelo, los vectores estarían en "mapas"
        #    distintos y la comparación no tendría sentido.
        #    embed() recibe y devuelve LISTAS: le pasamos [question]
        #    (una lista de un elemento) y con [0] tomamos el único vector.
        query_vector = self.embed([question])[0]

        # 2. Buscar en Pinecone los vectores más cercanos a la pregunta.
        return self.index.query(
            vector=query_vector,        # el punto de partida de la búsqueda
            top_k=top_k,                # como máximo, esta cantidad de resultados
            namespace=NAMESPACE,        # buscar SOLO en la partición "documentos"
            # Filtro rígido: "$eq" = "igual a". Los vectores cuya categoría
            # no sea la pedida ni siquiera compiten en la búsqueda.
            filter={"category": {"$eq": category}},
            # Sin esto, Pinecone devuelve solo id y score. Con True,
            # también devuelve los metadatos (incluido el texto original).
            include_metadata=True)


# ---------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------
def main():
    # Conectarse a Pinecone con la clave del archivo .env
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # El índice lo creó el Ejercicio 1. Si no existe, no tiene sentido
    # seguir: SystemExit corta el programa con un mensaje claro
    # (mejor que un error confuso más adelante).
    if INDEX_NAME not in pc.list_indexes().names():
        raise SystemExit(f"No existe '{INDEX_NAME}': corré primero el Ejercicio 1.")

    # Conectarse al índice y armar el pipeline.
    # OpenAI() sin argumentos busca solo la variable OPENAI_API_KEY.
    index = pc.Index(INDEX_NAME)
    pipeline = IngestionPipeline(index, OpenAI())

    # Los datos de prueba: 5 documentos (2 de legal, 2 de hr y 1 de finanzas).
    # Cada documento es un diccionario con su texto y sus etiquetas.
    # En un caso real, estos textos saldrían de PDFs o archivos ya cortados
    # en pedazos (chunks).
    docs = [
        {"text": "Contrato de servicios con el proveedor de hosting, vigente por 12 meses.",
         "category": "legal", "author": "User1"},
        {"text": "Acuerdo de confidencialidad (NDA) firmado con la agencia de diseño.",
         "category": "legal", "author": "User1"},
        {"text": "Política de licencias: 15 días de vacaciones y licencia por estudio.",
         "category": "hr", "author": "User2"},
        {"text": "Manual de onboarding para empleados nuevos del área de ventas.",
         "category": "hr", "author": "User2"},
        {"text": "Presupuesto anual 2026 con el detalle de gastos por departamento.",
         "category": "finanzas", "author": "User3"},
    ]

    # ---- PASO 1: INGESTA --------------------------------------------
    # batch_size=2 es MUY chico a propósito, para poder ver los lotes.
    # Con 5 documentos salen 3 lotes: 2 + 2 + 1.
    # En producción se usan lotes de 100 a 200 vectores.
    total = pipeline.process_and_upsert_batches(docs, batch_size=2)
    print(f"Total procesado: {total}")

    # Pinecone tarda unos segundos en dejar "buscables" los vectores nuevos.
    # Si buscáramos enseguida, podríamos recibir 0 resultados.
    time.sleep(10)

    # ---- PASO 2: BÚSQUEDA FILTRADA ----------------------------------
    # Fijate que la pregunta dice "acuerdos de confidencialidad", no "NDA":
    # la búsqueda es por SIGNIFICADO, no por palabras exactas.
    pregunta = "¿Qué acuerdos de confidencialidad tenemos firmados?"
    result = pipeline.search_by_category(pregunta, "legal")

    # result.matches es la lista de resultados, ordenada del más parecido
    # al menos parecido. Cada match trae:
    #     .score    → qué tan parecido es (con coseno: cuanto más alto, mejor)
    #     .id       → el id que le pusimos ("doc-2")
    #     .metadata → el diccionario de create_metadata (con el texto)
    # {match.score:.3f} muestra el número con 3 decimales.
    print(f"\nPregunta: {pregunta}")
    for match in result.matches:
        print(f"  {match.score:.3f} · {match.id} · {match.metadata['text']}")
    # Pedimos top_k=5, pero solo aparecen 2: son los únicos de "legal".

    # ---- PASO 3: ESTADÍSTICAS ---------------------------------------
    # Ahora el índice tiene DOS namespaces: "dev-environment" (3 vectores
    # del Ejercicio 1) y "documentos" (5 vectores de este ejercicio).
    print(index.describe_index_stats())


# Esta condición es verdadera solo cuando ejecutás el archivo directamente
# (python ejercicio2.py). Si otro archivo lo importa para usar la clase
# IngestionPipeline, main() NO se ejecuta sola.
if __name__ == "__main__":
    main()