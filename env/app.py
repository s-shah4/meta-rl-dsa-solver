import gradio as gr

def solve(problem):
    return "Model output will come here"

demo = gr.Interface(
    fn=solve,
    inputs="text",
    outputs="text",
    title="DSA Solver"
)

demo.launch()