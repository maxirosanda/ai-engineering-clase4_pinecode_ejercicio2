# =====================================================================
# EJERCICIO 2: ingesta en lotes con metadatos y búsqueda filtrada
# =====================================================================
# Continúa el Ejercicio 1: usa el MISMO índice ("mi-indice-serverless").
# Antes de correrlo (una sola vez), en la terminal:
#     pip install pinecone openai python-dotenv
# y en tu archivo .env:
#     PINECONE_API_KEY=...
#     OPENAI_API_KEY=...
# =====================================================================
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "mi-indice-serverless"          # el índice que creó el Ejercicio 1
NAMESPACE = "documentos"                     # otra partición, separada de "dev-environment"
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dimensiones: igual que el índice


# ---------------------------------------------------------------------
# EL PIPELINE DE INGESTA
# ---------------------------------------------------------------------
class IngestionPipeline:
    def __init__(self, index, openai_client: OpenAI):
        self.index = index
        self.openai = openai_client

    # Convierte una LISTA de textos en una lista de vectores.
    # Una sola llamada a OpenAI sirve para todo el lote.
    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]

    # La ÚNICA función que arma metadatos: así todos los vectores tienen
    # las mismas claves y el mismo formato (evita el "schema drift").
    def create_metadata(self, doc_text: str, category: str, author: str) -> Dict[str, Any]:
        return {
            "text": doc_text,              # el texto original: el LLM lo va a necesitar
            "category": category,
            "author": author,
            # isoformat() da un texto estándar: "2026-11-24T08:30:00.123456"
            "ingested_at": datetime.now().isoformat(),
            "char_count": len(doc_text),   # tamaño del texto en caracteres
        }

    def process_and_upsert_batches(self, documents: List[Dict], batch_size: int = 100) -> int:
        total = 0

        # range(0, 5, 2) da 0, 2, 4: la posición donde empieza cada lote
        for i in range(0, len(documents), batch_size):

            # 1. Tomar un trozo de la lista: desde i hasta i + batch_size.
            #    Si quedan menos documentos, el último lote sale más chico.
            batch = documents[i : i + batch_size]

            # 2. Embeddings REALES de todo el lote, en una sola llamada
            embeddings = self.embed([doc["text"] for doc in batch])

            # 3. Preparar los vectores del lote: id, values y metadata.
            #    El id es fijo ("doc-1", "doc-2"...): si corrés el programa
            #    otra vez, upsert los actualiza en vez de duplicarlos.
            vectors = [
                {
                    "id": f"doc-{i + j + 1}",
                    "values": embedding,
                    "metadata": self.create_metadata(
                        doc["text"], doc["category"], doc["author"]),
                }
                for j, (doc, embedding) in enumerate(zip(batch, embeddings))
            ]

            # 4. UNA llamada a upsert por lote (no una por documento)
            self.index.upsert(vectors=vectors, namespace=NAMESPACE)
            print(f"Lote subido: {len(vectors)} vectores")
            total += len(vectors)

        return total

    def search_by_category(self, question: str, category: str, top_k: int = 5):
        # La pregunta se convierte en vector con el MISMO modelo
        query_vector = self.embed([question])[0]

        # "$eq" = igual a. Solo se buscará entre vectores de esa categoría.
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE,
            filter={"category": {"$eq": category}},
            include_metadata=True)     # que devuelva también el texto


# ---------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------
def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # El índice lo creó el Ejercicio 1: si no existe, avisamos y salimos
    if INDEX_NAME not in pc.list_indexes().names():
        raise SystemExit(f"No existe '{INDEX_NAME}': corré primero el Ejercicio 1.")

    index = pc.Index(INDEX_NAME)
    pipeline = IngestionPipeline(index, OpenAI())   # OpenAI() lee OPENAI_API_KEY

    # 5 documentos: 2 de legal, 2 de hr y 1 de finanzas
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

    # 1. Ingesta de a 2: tienen que salir 3 lotes (2, 2 y 1)
    total = pipeline.process_and_upsert_batches(docs, batch_size=2)
    print(f"Total procesado: {total}")

    # Pinecone tarda unos segundos en dejar buscables los vectores nuevos
    time.sleep(10)

    # 2. Búsqueda filtrada: solo documentos de la categoría "legal"
    pregunta = "¿Qué acuerdos de confidencialidad tenemos firmados?"
    result = pipeline.search_by_category(pregunta, "legal")

    print(f"\nPregunta: {pregunta}")
    for match in result.matches:
        print(f"  {match.score:.3f} · {match.id} · {match.metadata['text']}")

    # 3. Estadísticas: ahora el índice tiene DOS namespaces
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()