from web.overwrites import postprocess
from web.utils import (
    convert_to_markdown,
    shared_state,
    reset_textbox,
    cancel_outputing,
    transfer_input,
    reset_state,
    delete_last_conversation
)
from web.presets import (
    small_and_beautiful_theme,
    title,
    description,
    description_top,
    CONCURRENT_COUNT
)
from sqlalchemy import create_engine
from backoff import on_exception, expo
import pandas as pd
from openai import OpenAI
import openai
import nltk
import gradio as gr
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
os.chdir(os.path.dirname(os.path.dirname(current_dir)))

# from doc_qa import DocQAPromptAdapter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

openai.api_key = "xxx"


def add_llm(model_name, api_base, models):
    """ 添加模型 """
    models = models or {}
    if model_name and api_base:
        models.update(
            {
                model_name: api_base
            }
        )
    choices = [m[0] for m in models.items()]
    return "", "", models, gr.Dropdown.update(choices=choices, value=choices[0] if choices else None)


def set_openai_env(api_base):
    """ 配置接口地址 """
    openai.api_base = api_base
    # doc_adapter.embeddings.openai_api_base = api_base


def get_file_list():
    """ 获取文件列表 """
    if not os.path.exists("doc_store"):
        return []
    return os.listdir("doc_store")


file_list = get_file_list()


@on_exception(expo, openai.RateLimitError, max_tries=5)
def chat_completions_create(client, params):
    """ chat接口 """
    return client.chat.completions.create(**params)


@on_exception(expo, openai.RateLimitError, max_tries=5)
def completions_create(client, params):
    return client.chat.completions.create(**params)


def predict(
    model_name,
    models,
    combined_input,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    single_turn,
):

    api_base = models.get(model_name)
    client = OpenAI(api_key=openai.api_key, base_url=api_base)

    if combined_input == "":
        yield chatbot, history, "Empty context.", None
        return

    if history is None:
        history = []

    messages = []

    system_prompt, text = combined_input.split("\nUser: ")
    system_prompt = system_prompt.replace("System: ", "")

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    if not single_turn:
        for h in history[-memory_k:]:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": h[0]
                    },
                    {
                        "role": "assistant",
                        "content": h[1]
                    }
                ]
            )

    messages.append(
        {
            "role": "user",
            "content": text
        }
    )

    params = dict(
        stream=True,
        messages=messages,
        model=model_name,
        top_p=top_p,
        temperature=temperature,
        max_tokens=int(max_tokens)
    )
    res = chat_completions_create(client, params)

    x = ""

    for openai_object in res:
        try:
            if openai_object.choices[0].delta.content is not None:
                x += openai_object.choices[0].delta.content
        except:
            continue

        a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
            [text, convert_to_markdown(x)]
        ], history + [[text, x]]

        yield a, b, "Generating...", None

    if shared_state.interrupted:
        shared_state.recover()
        try:
            yield a, b, "Stop: Success", None
            return
        except:
            pass


def retry(
    model_name,
    models,
    combined_input,
    chatbot,
    history,
    top_p,
    temperature,
    max_tokens,
    memory_k,
    single_turn,
):

    logging.info(f"model_name: {model_name}")
    logging.info(f"history: {history}")
    logging.info(f"top_p: {top_p}")
    logging.info(f"temperature: {temperature}")
    logging.info(f"max_tokens: {max_tokens}")
    logging.info(f"memory_k: {memory_k}")
    logging.info(f"single_turn: {single_turn}")
    logging.info("Retry...")

    if len(history) == 0:
        yield chatbot, history, "Empty context."
        return

    chatbot.pop()
    last_interaction = history.pop()
    user_input = last_interaction[0]
    system_response = combined_input.split(
        '\nUser: ')[0].replace('System: ', '')
    combined_input = f"System: {system_response}\nUser: {user_input}"

    for result in predict(
        model_name,
        models,
        combined_input,
        chatbot,
        history,
        top_p,
        temperature,
        max_tokens,
        memory_k,
        single_turn,
    ):
        yield result


gr.Chatbot.postprocess = postprocess

with open("/home/sunjinf/github_projet/deployment/web/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")
    system_prompt = gr.State("")

    gr.Markdown("# OpenLLM Test 🚀")

    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")

    gr.Markdown(description_top)

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=12):
                with gr.Row():
                    chatbot = gr.Chatbot(elem_id="Simon's chatbot", height=500)
                with gr.Row():
                    with gr.Column(scale=12):
                        system_input = gr.Textbox(
                            show_label=False, lines=2, placeholder="Enter system prompt"
                        )
                        user_input = gr.Textbox(
                            show_label=False, lines=5, placeholder="Enter text"
                        )
                    with gr.Column(min_width=70, scale=1):
                        submitBtn = gr.Button("☝️发送")
                        cancelBtn = gr.Button("🙅停止")
                with gr.Row():
                    emptyBtn = gr.Button("🧹 新的对话")
                    retryBtn = gr.Button("🔄 重新生成")
                    delLastBtn = gr.Button("🗑️ 删除最旧对话")

            with gr.Column():
                with gr.Column(min_width=60, scale=1):
                    with gr.Tab(label="模型"):
                        with gr.Accordion(open=False, label="🧮所有模型配置"):
                            models = gr.Json(
                                value={
                                    "Yi-1.5-34B-Chat-16K-GPTQ-Int4": "http://10.0.34.61:9012/v1",
                                    "Qwen2-72B-Chat-GPTQ-Int4": "http://10.0.34.61:9010/v1",
                                    "Qwen1.5-14B-Chat": "http://10.0.34.61:7893/v1",
                                    "Baichuan2-13B-chat": "http://10.0.34.61:9006/v1",
                                    "Qwen2-7B-Chat": "http://10.0.34.61:9004/v1",
                                    "LLama-3-SQLcoder-8b": "http://10.0.34.61:9015/v1",
                                    "公文正文模型": "http://10.0.34.61:6600/v1",
                                    "公文大纲模型": "http://10.0.34.61:9008/v1",

                                }
                            )
                        single_turn = gr.Checkbox(label="使用单轮对话", value=False)
                        select_model = gr.Dropdown(
                            choices=[m[0] for m in models.value.items()] if models.value else [
                            ],
                            value=[m[0] for m in models.value.items(
                            )][0] if models.value else None,
                            label="选择模型",
                            interactive=True,
                        )
                    with gr.Tab(label="🔢参数"):
                        top_p = gr.Slider(
                            minimum=0,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=1,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        max_tokens = gr.Number(
                            value=12000,
                            step=512,
                            maximum=200000,
                            minimum=512,
                            interactive=True,
                            label="Max Generation Tokens",
                        )
                        memory_k = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=5,
                            step=1,
                            interactive=True,
                            label="Max Memory Window Size",
                        )
                        chunk_size = gr.Slider(
                            minimum=100,
                            maximum=1000,
                            value=200,
                            step=100,
                            interactive=True,
                            label="Chunk Size",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=10,
                            interactive=True,
                            label="Chunk Overlap",
                        )
                with gr.Tab(label="添加模型"):
                    model_name_input = gr.Textbox(label="模型名称")
                    api_base_input = gr.Textbox(label="API Base URL")
                    add_model_btn = gr.Button("➕添加模型")
                    add_model_btn.click(add_llm, [model_name_input, api_base_input, models], [
                        model_name_input, api_base_input, models, select_model])

    gr.Markdown(description)

    def transfer_input(system_input, user_input):
        # Combine system and user input for prediction
        combined_input = f"System: {system_input}\nUser: {user_input}"
        return (
            combined_input,
            gr.update(value=""),
            gr.update(value=""),
            gr.Button.update(visible=True),
            gr.Button.update(visible=True),
        )

    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[system_input, user_input],
        outputs=[user_question,
                 user_input, system_input, submitBtn, cancelBtn],
        show_progress=True,
    )

    predict_args = dict(
        fn=predict,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            single_turn,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    retry_args = dict(
        fn=retry,
        inputs=[
            select_model,
            models,
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_tokens,
            memory_k,
            single_turn,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(
        fn=reset_textbox,
        inputs=[],
        outputs=[user_input, status_display]
    )

    user_input.submit(**transfer_input_args).then(**predict_args)
    system_input.submit(**transfer_input_args).then(**predict_args)
    submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    retryBtn.click(**retry_args)
    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )

demo.title = "Chatbot Test"

if __name__ == "__main__":
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
        server_name="10.0.34.62", server_port=7835, share=True)
