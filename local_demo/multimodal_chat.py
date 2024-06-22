import gradio as gr
import os
import json
from datetime import datetime
import hashlib
import argparse
from PIL import Image

from theme_dropdown import create_theme_dropdown  # noqa: F401
from constants import (
    html_header,
    tos_markdown,
    learn_more_markdown,
    bibtext,
)
dropdown, js = create_theme_dropdown()

from longva_backend import LongVA
longva = LongVA(pretrained="lmms-lab/LongVA-7B-DPO", model_name="llava_qwen", device_map="auto", device="cuda")


def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:6]


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message, video_input=None):
    if video_input is not None and video_input != "" and len(message["files"]) == 0:
        history.append(((video_input,), None))
    else:
        for x in message["files"]:
            history.append(((x,), None))

    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def http_bot(
    video_input,
    state,
    sample_frames=16,
    temperature=0.2,
    max_new_tokens=8192,
    top_p=1.0,
):
    try:
        visual_count = 0
        conv_count = 0
        prev_conv = []
        last_visual_index = -1
        for idx, x in enumerate(state):
            if type(x[0]) == tuple:
                visual_count += 1
                image_path = x[0][0]
                last_visual_index = idx
            elif type(x[0]) == str and type(x[1]) == str:
                conv_count += 1
                prev_conv.append(x)

        if visual_count == 1 and video_input is not None:
            image_path = video_input
            task_type = "video"
        elif visual_count == 1 and video_input is None and type(state[0][0]) == tuple:
            task_type = "image"
        elif visual_count == 0:
            image_path = ""
            task_type = "text"
        elif visual_count > 1:
            print(f"Visual count: {visual_count}")
            visuals = state[last_visual_index][0]
            state = state[last_visual_index:]
            prev_conv = []
            if visuals[0].split(".")[-1] in ["mp4", "mov", "avi", "mp3", "wav", "mpga", "mpg", "mpeg"]:
                task_type = "video"
            else:
                task_type = "image"

        prompt = state[-1][0]

        if task_type != "text" and not os.path.exists(image_path):
            state[-1][1] = "The conversation is not correctly processed. Please try again."
            return state

        if task_type != "text":
            print(f"Processing Visual: {image_path}")
            print(f"Processing Question: {prompt}")
        else:
            print(f"Processing Question (text): {prompt}")

        try:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": False,
                "top_p": top_p,
            }
            state[-1][1] = ""

            if task_type == "text":
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [],
                    "context": prompt,
                    "task_type": task_type,
                }
                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            elif image_path.split(".")[-1] in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                task_type = "image"
                # stream output
                image = Image.open(image_path).convert("RGB")
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [image],
                    "context": prompt,
                    "task_type": task_type,
                }

                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            elif image_path.split(".")[-1] in ["mp4", "mov", "avi", "mp3", "wav", "mpga", "mpg", "mpeg"]:
                task_type = "video"
                request = {
                    "prev_conv": prev_conv,
                    "visuals": [image_path],
                    "context": prompt,
                    "task_type": task_type,
                }
                gen_kwargs["sample_frames"] = sample_frames

                prev = 0
                for x in longva.stream_generate_until(request, gen_kwargs):
                    output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
                    print(output[prev:], end="", flush=True)
                    state[-1][1] += output[prev:]
                    prev = len(output)
                    yield state

            else:
                state[-1][1] = "Image format is not supported. Please upload a valid image file."
                yield state
        except Exception as e:
            raise e

    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LongVA-7B-DPO", help="Model name")
    parser.add_argument("--temperature", default="0", help="Temperature")
    parser.add_argument("--max_new_tokens", default="8192", help="Max new tokens")
    args = parser.parse_args()

    PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
    LOGDIR = f"{PARENT_FOLDER}/logs"
    print(PARENT_FOLDER)
    print(LOGDIR)

    chatbot = gr.Chatbot(
        [],
        label=f"Model: {args.model_name}",
        elem_id="chatbot",
        bubble_full_width=False,
        height=700,
        avatar_images=(
            (
                os.path.join(
                    os.path.dirname(__file__), f"{PARENT_FOLDER}/assets/user_logo.png"
                )
            ),
            (
                os.path.join(
                    os.path.dirname(__file__),
                    f"{PARENT_FOLDER}/assets/assistant_logo.png",
                )
            ),
        ),
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image", "video"],
        placeholder="Enter message or upload file...",
        show_label=False,
        max_lines=10000,
    )

    with gr.Blocks(
        theme="finlaymacklon/smooth_slate",
        title="LongVA Multimodal Chat from LMMs-Lab",
        css=".message-wrap.svelte-1lcyrx4>div.svelte-1lcyrx4  img {min-width: 50px}",
    ) as demo:
        gr.HTML(html_header)
        # gr.Markdown(title_markdown)
        # gr.Markdown(subtitle_markdown)

        models = ["LongVA-7B-DPO"]
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=models,
                    value=models[0] if len(models) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                )

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    sample_frames = gr.Slider(
                        minimum=0,
                        maximum=256,
                        value=16,
                        step=4,
                        interactive=True,
                        label="Sample Frames",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=8192,
                        value=1024,
                        step=256,
                        interactive=True,
                        label="Max output tokens",
                    )

                video = gr.Video(label="Input Video", visible=False)
                gr.Examples(
                    examples=[
                        [
                            f"{PARENT_FOLDER}/assets/dc_demo.mp4",
                            {
                                "text": "What's the video about?",
                            },
                        ],
                        [
                            f"{PARENT_FOLDER}/assets/water.mp4",
                            {
                                "text": "Why does thie main cook the ice cube?",
                            },
                        ],
                        [
                            f"{PARENT_FOLDER}/assets/jobs.mp4",
                            {
                                "text": "Please conclude this new product launch event.",
                            },
                        ],
                    ],
                    inputs=[video, chat_input],
                )
                gr.Examples(
                    examples=[
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_05.jpg",
                            ],
                            "text": "この猫の目の大きさは、どのような理由で他の猫と比べて特に大きく見えますか？",
                        },
                        {
                            "files": [
                                f"{PARENT_FOLDER}/assets/user_example_06.jpg",
                            ],
                            "text": "Write the content of this table in a Notion format?",
                        },
                        # {
                        #     "files": [
                        #         f"{PARENT_FOLDER}/assets/user_example_10.png",
                        #     ],
                        #     "text": "Here's a design for blogging website. Provide the working source code for the website using HTML, CSS and JavaScript as required.",
                        # },
                    ],
                    inputs=[chat_input],
                )
                with gr.Accordion("More Examples", open=False) as more_examples_row:
                    gr.Examples(
                        examples=[
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/otter_books.jpg",
                                ],
                                "text": "Why these two animals are reading books?",
                            },
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/user_example_09.jpg",
                                ],
                                "text": "请针对于这幅画写一首中文古诗。",
                            },
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/white_cat_smile.jpg",
                                ],
                                "text": "Why this cat smile?",
                            },
                            {
                                "files": [
                                    f"{PARENT_FOLDER}/assets/user_example_07.jpg",
                                ],
                                "text": "这个是什么猫？",
                            },
                        ],
                        inputs=[chat_input],
                    )
            with gr.Column(scale=9):
                chatbot.render()
                chat_input.render()

                chat_msg = chat_input.submit(
                    add_message, [chatbot, chat_input, video], [chatbot, chat_input]
                )
                bot_msg = chat_msg.then(
                    http_bot,
                    inputs=[
                        video,
                        chatbot,
                        sample_frames,
                        temperature,
                        max_output_tokens,
                        top_p,
                    ],
                    outputs=[chatbot],
                    api_name="bot_response",
                )
                bot_msg.then(
                    lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                )
                bot_msg.then(lambda: video, None, [video])

                chatbot.like(print_like_dislike, None, None)

                with gr.Row():
                    clear_btn = gr.ClearButton(chatbot, chat_input, chat_msg, bot_msg)
                    clear_btn.click(
                        lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                    ).then(
                        lambda: video, None, [video]
                    )
                    
                    submit_btn = gr.Button("Send", chat_msg)
                    submit_btn.click(
                        add_message, [chatbot, chat_input, video], [chatbot, chat_input]
                    ).then(
                        http_bot,
                        inputs=[
                            video,
                            chatbot,
                            sample_frames,
                            temperature,
                            max_output_tokens,
                            top_p,
                        ],
                        outputs=[chatbot],
                        api_name="bot_response",
                    ).then(
                        lambda: gr.MultimodalTextbox(interactive=True),
                        None,
                        [chat_input],
                    ).then(
                        lambda: video, None, [video]
                    )

        gr.Markdown(bibtext)
        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)

    demo.queue(max_size=128)
    demo.launch(max_threads=8, share=False, server_port=8000, favicon_path=f"{PARENT_FOLDER}/assets/favicon.ico")