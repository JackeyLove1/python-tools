import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
import os

import fitz
from PIL import Image
with gr.Blocks() as demo:
    # Create a Gradio block

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
                    show_label=False,
                    interactive=True
                ).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')

        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=680)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter"
            ).style(container=False)

        with gr.Column(scale=0.15):
            submit_btn = gr.Button('Submit')

        with gr.Column(scale=0.15):
            btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"]).style()

    # Set up event handlers

    # Event handler for submitting the OpenAI API key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])

    # Event handler for changing the API key
    change_api_key.click(fn=enable_api_box, outputs=[api_key])

    # Event handler for uploading a PDF
    btn.upload(fn=render_first, inputs=[btn], outputs=[show_img])

    # Event handler for submitting text and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )
enable_box = gr.Textbox.update(value=None,placeholder= 'Upload your OpenAI API key',
                               interactive=True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set',interactive=False)


def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box


def enable_api_box():
    return enable_box


def process_file(file):
    # raise an error if API key is not provided
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file.name)
    documents = loader.load()

    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()

    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and PDF search retriever
    pdfsearch = Chroma.from_documents(documents, embeddings, )

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                  retriever=
                                                  pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True, )
    return chain


def generate_response(history, query, btn):
    global COUNT, N, chat_history

    # Check if a PDF file is uploaded
    if not btn:
        raise gr.Error(message='Upload a PDF')

    # Initialize the conversation chain only once
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1

    # Generate a response using the conversation chain
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)

    # Update the chat history with the query and its corresponding answer
    chat_history += [(query, result["answer"])]

    # Retrieve the page number from the source document
    N = list(result['source_documents'][0])[1][1]['page']

    # Append each character of the answer to the last message in the history
    for char in result['answer']:
        history[-1][-1] += char

        # Yield the updated history and an empty string
        yield history, ''

