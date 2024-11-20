import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json


# Set up environment variables for API keys
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
os.environ['PINECONE_API_KEY'] = 'YOUR_PINECONE_API_KEY'

# Define the index name and embeddings
index_name = "YOUR_INDEX_NAME_IN_PINECONE"
embeddings = OpenAIEmbeddings()

# Load the PDF document
loader = PyPDFLoader(r"YOUR_PDFFILE_PATH")
documents = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create vector store from documents
vectorstore = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)

# Initialize OpenAI language model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Define the retriever to use Pinecone vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


# Define a query function to get the answer for a given question
def query_pdf(question):
    # Run the QA chain using invoke method
    response = qa_chain.invoke({"query": question})

    # Check if response is a dictionary and properly format it
    if isinstance(response, dict):
        # Assuming the answer is stored in a field named 'result' or similar
        answer = response.get("result", "Answer not found.")
    else:
        answer = response
    
    # Format the answer neatly
    tidy_answer = json.dumps(answer, indent=2) if isinstance(answer, dict) else str(answer).strip()

    return tidy_answer

# Example query
question = "What are the main contributions of the paper?"
answer = query_pdf(question)
print(f"Question: {question}\nAnswer:\n{answer}")



