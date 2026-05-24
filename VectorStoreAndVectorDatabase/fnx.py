import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models.base import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import re
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from typing import Any
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

llm = ChatOllama(model="gemma3:12b")
embedding = OllamaEmbeddings(model="qwen3-embedding:8b")

persist_directory = "./vectorstore"

fnx_doc = []
chain = None
custom_prompt = """
        You are a professional customer service representative for Phoenix Insurance Company.
        ***Always respond in Hebrew regardless of the language used in the question***.

        Your goal is to provide accurate information about the company's policies.

        Guidelines:
        * Use a professional, patient, and respectful tone.
        * If asked analytical questions, respond that you are not an analytics tool — DO NOT RETURN ANY DATA!
        * If information is missing or not found in the system — state it clearly.
        * Do not fabricate information that does not exist in the provided data.
        * If multiple similar results are found — ask for additional details to identify the correct one.
        * If the question is unrelated to Phoenix or the data in the system — politely respond that you cannot assist with that topic.
        * Phrase answers in a natural, human way — not as a raw technical data list.

        Example response style:
        * "Your policy is active until 11/05/2026..."
        * "The agent handling your policy is SMART הפניקס..."
        * "No policy was found matching the provided details..."

        ***Always respond in Hebrew***.

        Context: {context}
    """


def init():
    load_data()
    vs = build_vector_db()
    hybrid_retriever = build_retriever(vs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", custom_prompt),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    global  chain
    chain = create_retrieval_chain(hybrid_retriever, document_chain)


def load_data():
    global fnx_doc
    fnx_doc = []

    try:
        policies_table = pd.read_excel("../DataIngestParsing/data/csv_excel/policies_table.xlsx")
        users_table = pd.read_excel("../DataIngestParsing/data/csv_excel/users_table.xlsx")
        cars_tabel = pd.read_excel("../DataIngestParsing/data/csv_excel/cars_table.xlsx")

        for _, row in policies_table.iterrows():
            user_matched = users_table[users_table["Id"] == row["CustomerId"]]
            user_row = user_matched.iloc[0] if not user_matched.empty else pd.Series(dtype=object)

            car_matched = cars_tabel[cars_tabel["PolicyId"] == row["Id"]]
            if car_matched.empty:
                cars_text = "אין לפוליסה זו רכבים"
            else:
                car_lines = []
                for _, car_row in car_matched.iterrows():
                    car_lines.append(
                        f'* {car_row.get("ModelName", "")} שנת {car_row.get("ManufactureYear", "")} מספר {car_row.get("CarNumber", "")} - בעל הרכב {car_row.get("CarOwnerDetails", "")} - תעריף לקילומטר {car_row.get("KmPremium", "")} ש"ח - הרכב מבוטח בביטוח {car_row.get("InsuranceDesc", '')} - שווי רכב משוער {car_row.get("CarPrice", '')}'
                    )
                    cars_text = "\n".join(car_lines)

            text = f"""
            פוליסה מספר {row.get('PolicyNumber', '')},
            שנת חיתום: {row.get("ShnatChitum", '')}
            ענף: {row.get("Anaf", '')}
            תאריך תחילת הפוליסה: {row.get("PolicyStartDate", '')}
            תאריך סיום הפוליסה: {row.get("PolicyEndDate", '')}
            מזהה לקוח: {row.get('CustomerId', '')}

            בעל הפוליסה: {user_row.get('FirstName', '')} {user_row.get('LastName', '')}
            תעודת זהות: {user_row.get('GovId', '')}

            כתובת בעל הפוליסה: 
             {user_row.get('CityDesc', '')}, {user_row.get('StreetDesc', '')} {user_row.get('HouseNumber', '')}

            תאריך לידה: {user_row.get('BirthdayDate', '')}
            שנת הוצאת רישיון: {user_row.get('YearlicenseIssued', '')}
            כתובת אימייל: {user_row.get('Email', '')}

            הפוליסה מנוהלת על ידי הסוכן: {row.get('AgentName', '')}
            מספר סוכן: {row.get('AgentId', '')}
            הפוליסה נוצרה על ידי: {row.get('CreatedBy', '')}

            סטטוס הפוליסה: {"לא פעילה" if row.get("IsActive") == 0 else "פעילה"}.
            מספר התביעות שהיו לפוליסה: {row.get("NumberOfClaims", '')}
            מספר השלילות שהיו לפוליסה: {row.get("NumberOfDenials", '')}

            רכבים: 
             {cars_text}
            """

            # Build metadata
            metadata = {
                "מספר פוליסה מלא": row.get('PolicyId', ''),
                "מספר הפוליסה": row.get('PolicyNumber', ''),
                "govId": str(user_row.get("GovId", "")),
            }

            fnx_doc.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        print(f"The embedding data will look like that: {fnx_doc[20]}")
    except Exception as e:
        print(f"Error loading PyPDF: {e}")


def build_vector_db():
    os.makedirs(persist_directory, exist_ok=True)
    vs = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name="core"
    )

    if vs._collection.count() == 0:
        vs.add_documents(fnx_doc)
        print(f"Populated vector store with {vs._collection.count()} vectors")
    else:
        print(f"Loaded existing vector store with {vs._collection.count()} vectors")

    return vs


def build_retriever(vector_store):
    # BM25 retriever (Exact mach for names - names are not embed great so the semantic search is weak)
    bm25_retriever = BM25Retriever.from_documents(fnx_doc, k=3)
    # Dense retriever - Regular similarity search
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 80/50 blend — tune weights based on your results
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.8, 0.2]
    )

    class HybridRetriever(BaseRetriever):
        vector_store: Any = Field(...)
        ensemble: Any = Field(...)
        k: int = Field(default=3)

        def _get_relevant_documents(self, query: str) -> list[Document]:
            # 1. Exact policy number / id lookup via metadata
            policy_match = re.search(r'\b(\d{6,10})\b', query)
            id_match = re.search(r'\b(\d{9})\b', query)

            if id_match:
                id_number = int(id_match.group(1))
                results = self.vector_store.get(where={"govId": id_number})
                docs = [
                    Document(page_content=pc, metadata=meta)
                    for pc, meta in zip(results["documents"], results["metadatas"])
                ]
                if docs:
                    return docs

            if policy_match:
                policy_number = int(policy_match.group(1))
                results = self.vector_store.get(where={"הסילופה רפסמ": policy_number})
                docs = [
                    Document(page_content=pc, metadata=meta)
                    for pc, meta in zip(results["documents"], results["metadatas"])
                ]
                if docs:
                    return docs

            # 2. BM25 + semantic ensemble for everything else (names, questions, etc.)
            return self.ensemble.invoke(query)

    retriever = HybridRetriever(vector_store=vector_store, ensemble=ensemble_retriever)
    return retriever


def query(prompt: str):
    # Method 1 - Single shot
    # response = chain.invoke({"input": prompt})
    # # print(response.get("answer"))
    # return response.get("answer", "")

    # Method 2 - Streaming
    partial = ""
    for chunk in chain.stream({ "input": prompt }):
        if token := chunk.get("answer"):
            partial += token
            yield  partial


init()