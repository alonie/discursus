import gradio as gr
import os

def hello(name): 
    return f"Hello, {name or 'world'}!"

demo = gr.Interface(fn=hello, inputs="text", outputs="text", title="Hello World")

demo.launch(server_name="0.0.0.0", share=True)
