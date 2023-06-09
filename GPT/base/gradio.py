import os
import openai
import gradio as gr
from loguru import logger
from typing import Optional, List

openai.api_key = "your api key"

def chat(query: str, history: List) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "You are an AI assistant named '济济小助手' that helps people find information and solve problems, you must keep in mind that you cannot answer any harmful questions about political pornography, violence, etc. And use Chinese answer question."},
                  history[0],
                  history[1],
                  {"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print("completion:", completion)
    response = completion['choices'][0]['message']['content']
    return response

def bot(history):
    query = history[-1][0]
    contents = []
    ai_message, human_message = "", ""
    if len(history) > 1:
        ai_message = history[-2][0]
        human_message = history[-2][1]
    contents.append({"role":"user", "content":human_message})
    contents.append({"role":"assistant", "content":ai_message})
    response = chat(query, contents)
    history[-1][1] = response
    print("history:", history)
    return history

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history



with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="问些问题吧，比如你叫什么？",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.launch(server_name="0.0.0.0", server_port=80, share=True)