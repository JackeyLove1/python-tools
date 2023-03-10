import gradio as gr

def add_text(state, text):
    state = state + [(text, text + "?")]
    return state, state


with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                container=False)
        '''
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("🖼️", file_types=["image"])
        '''
    txt.submit(add_text, [state, txt], [state, chatbot])
    txt.submit(lambda: "", None, txt)
    # btn.upload(add_image, [state, btn], [state, chatbot])

demo.launch(server_port=8088)