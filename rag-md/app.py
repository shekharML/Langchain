from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
from vector import md_rag

ollama_url = os.environ.get("OLLAMA_API_BASE")
model = OllamaLLM(model="deepseek-r1:7b", base_url=ollama_url)

template = """
you are an expert in answering technical questions and you will get answers from Markdown files.
Here are some relevant documents: {documents}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

md_files = "./markdown_files"
retriever = md_rag(
    source_dir=md_files, db_path="chroma_md_db", collection_name="tech_files"
)

print("Retriever Ready.")

while True:
    print("\n-----------------------------------------")
    question = input("Ask your question (q to quit)")
    print("\n\n")
    if question.lower == "q":
        break

    docs = retriever.invoke(question)
    result = chain.invoke({"documents": docs, "question": question})
    print(f"AI:{result}")
