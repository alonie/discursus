import gradio as gr
import os

ICON_PATH = "Discursus_logo_23Sep25.png"  # Place Discursus_logo_23Sep25.png in the same directory as app.py

def hello(name): 
    return f"Hello, {name or 'world'}!"

with gr.Blocks(title="Discursus") as demo:
    with gr.Row():
        gr.Image(value=ICON_PATH, show_download_button=False, interactive=False, show_label=False)
    gr.Interface(fn=hello, inputs="text", outputs="text")

#demo = gr.Interface(fn=hello, inputs="text", outputs="text", title="Hello World")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)),
        share=True
    )
