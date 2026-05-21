import gradio as gr
from fnx import query

gr.Interface(
    fn=query,
    inputs=gr.Textbox(label="שאלה", placeholder="למשל: מה הכתובת של בעל הפוליסה 1029855?", rtl=True),
    outputs=gr.Textbox(label="תשובה", rtl=True),
    title="פוליסומט - הפניקס צעיר",
    description="שירות לקוחות",
    flagging_mode="never",
).launch(footer_links=[])