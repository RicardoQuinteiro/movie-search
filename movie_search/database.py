import json
from pathlib import Path
from typing import List, Union

from haystack import Document
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter


def load_database(db_file: Union[str, Path]) -> InMemoryDocumentStore:
    doc_store = InMemoryDocumentStore.load_from_disk(db_file)
    return doc_store


def load_document(file: Union[str, Path]):
    with open(Path(file), "r") as jfile:
        doc = json.load(jfile)
    return doc


def create_document_store(
    docs: List[Document],
    document_store: InMemoryDocumentStore,
    device: str = "cpu",
    model: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    save_file: Union[str, Path] = None,
):

    document_splitter = DocumentSplitter(split_by="word", split_length=512, split_overlap=32)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=model,
        device=ComponentDevice.from_str(device),
        batch_size=batch_size,
    )
    document_writer = DocumentWriter(document_store)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("document_splitter", document_splitter)
    indexing_pipeline.add_component("document_embedder", document_embedder)
    indexing_pipeline.add_component("document_writer", document_writer)

    indexing_pipeline.connect("document_splitter", "document_embedder")
    indexing_pipeline.connect("document_embedder", "document_writer")

    print("Embedding docs...")
    indexing_pipeline.run({"document_splitter": {"documents": docs}})

    if save_file:
        document_store.save_to_disk(save_file)

    return document_store


def embed_movie_db(
    db_folder: Union[str, Path],
    save_file: Union[str, Path],
    device: str = "cpu",
    batch_size: int = 32,
):
    list_of_files = [file for file in Path(db_folder).iterdir() if file.suffix == ".json"]
    docs = []

    print("Creating docs...")
    for file in tqdm(list_of_files):
        movie_info = load_document(file)
        doc = Document(
            content=str(movie_info),
            meta=movie_info,
        )
        docs.append(doc)

    document_store = InMemoryDocumentStore()
    document_store = create_document_store(
        docs=docs,
        document_store=document_store,
        save_file=save_file,
        batch_size=batch_size,
        device=device,
    )

    
    
    

