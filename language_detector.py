import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceEndpoint
import gradio as gr


load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_result(result: str) -> str:
    if "I'm not sure" in result:
        return "I'm not sure"
    result = result.strip().replace("\n","").replace(".","").replace("Answer", "").replace("Human", "").replace("Text", "").replace(":", "").strip()
    if result:
        result = result.split()[0]
    return result


def rag_chat_stream(message, history):
    buffer = ""
    for token in rag_chain.stream(message):
        buffer += token
        yield buffer


if __name__ == "__main__":

    # Initialize LLM
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
        repo_id=model,
        temperature=0.1,
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )
    # RAG prompt template
    template = """
        You are an AI language detection assistant.
        Here are some examples of Text and its language in one word:

        {context}

        Provide the language of the given text. Answer ONLY with one word. Answer ONLY in the Latin script.
        If you do not know the language, just say "I'm not sure". Don't try to make up an answer.
        Text:

        {input}
        
        Language Identified:
        """
    # Prompt
    custom_rag_prompt = PromptTemplate.from_template(template)
    # Creating a chain using LCEL
    # Passing input using RunnablePassthrough
    # Formatting the output of the vector store search by format_docs()
    # Formatting the result by format_result()
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "input": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | format_result
    )

    # Start the Gradio Chat Interface
    try:
        gr.ChatInterface(
            fn=rag_chat_stream,
            chatbot=gr.Chatbot(
                value=[(None, "Welcome 👋. I am a Language Detector. I will output the language of the text in Latin script. Please do not type any instructions, just input the text for which you want the language to be identified.")]),
            examples=[
                "На четири велика рељефа посетиоци ће моћи да виде угравиране сцене из Немањиног живота и српске историје",
                "الوصول لأعلى مراتب نشير هنا كما أشار التوجه أستاذ توفيق إلى قضية ورئاسة الدولة تحق لكل مواطن سوري في سوريا المستقبل",
                "Vì vậy, tôi nghe nói, cô ấy thừa nhận bằng một giọng nhỏ.",
                "A z dymu wyszła szarańcza na ziemię i dano jej moc, jaką mają skorpiony ziemskie",
                "Um marco subvalorizado e com altas taxas de juro foi prejudicial"
                ],
            title="Language Detector"
        ).queue().launch(share=True, debug=True)
    except Exception as exc:
        print(f"Session ended. Exception: {exc}")
