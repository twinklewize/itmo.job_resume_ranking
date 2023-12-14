import gradio as gr
import ranking

def rank_interface(model_choice, description, *items):
    items_list = [item for item in items if item]
    if model_choice == 'BERT':
        ranked_items = ranking.rank_items_bert(description, items_list)
    elif model_choice == 'Doc2Vec':
        ranked_items = ranking.rank_items_doc2vec(description, items_list, threshold=0.3)
    else:  # Doc2Vec + BERT
        ranked_items = ranking.rank_items_doc2vec_bert(description, items_list)

    output_html = "<ul>"
    for item, similarity in ranked_items:
        output_html += f"<li><strong>{similarity}</strong>: {item}</li>"
    output_html += "</ul>"
    return output_html


max_textboxes = 10

def variable_outputs(k):
    k = int(k)
    return [gr.Textbox(label=f"Текст {i+1}", visible=True, max_lines=5) for i in range(k)] + \
           [gr.Textbox(visible=False) for i in range(k, max_textboxes)]

with gr.Blocks() as demo:
    gr.Label("Resume Job Ranking")
    with gr.Row():
        model_selection = gr.Dropdown(
            choices=['Doc2Vec', 'BERT', 'Doc2Vec + BERT'],
            value='Doc2Vec',
            label="Выберите модель"
        )
        s = gr.Slider(1, max_textboxes, value=3, step=1, label="Количество видимых полей для ввода:")

    description_input = gr.Textbox(
        label="Описание",
        placeholder="Введите описание вакансии или резюме здесь...",
        lines=5
    )

    textboxes = [gr.Textbox(label=f"Текст {i+1}", visible=i < 3, max_lines=5) for i in range(max_textboxes)]
    s.change(variable_outputs, s, textboxes)

    submit_button = gr.Button("Отправить")
    output = gr.HTML(label="Результат")

    submit_button.click(
        fn=rank_interface, 
        inputs=[model_selection, description_input] + textboxes,
        outputs=output
    )

demo.launch()