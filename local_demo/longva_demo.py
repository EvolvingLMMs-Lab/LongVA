import copy
import math
import warnings
import json
import os

from datetime import timedelta
from typing import List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from transformers import AutoConfig

from transformers import TextIteratorStreamer
from threading import Thread

import requests

torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")
from loguru import logger as eval_logger

try:
    from longva.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
    )
    from longva.conversation import conv_templates, SeparatorStyle
    from longva.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from longva.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug(
        "LongVA is not installed. Please install LongVA to use this model.\nError: %s" % e
    )

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


class LongVA:
    """
    LongVA Model
    """

    def __init__(
        self,
        pretrained: str = "LongVa/LongVA-7B",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "average",
        token_strategy: Optional[
            str
        ] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame in the context
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        model_name = (
            model_name
            if model_name is not None
            else get_model_name_from_path(pretrained)
        )

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = (
                load_pretrained_model(
                    pretrained,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert (
            self.batch_size_per_gpu == 1
        ), "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                eval_logger.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0"
                )

            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(
                f"Using {accelerator.num_processes} devices with tensor parallelism"
            )
            self._rank = 0
            self._word_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        print(f"spare_frames: {spare_frames.shape}")
        return spare_frames  # (frames, height, width, channels)

    def stream_generate_until(self, requests: dict, gen_kwargs: dict) -> List[str]:

        question_input = []

        visuals = requests["visuals"]
        context = requests["context"]
        task_type = requests["task_type"]

        if task_type == "text":
            image_tensor = None

        # encode, pad, and truncate contexts for this batch
        elif task_type == "image":  # For image task
            image_tensor = process_images(visuals, self._image_processor, self._config)
            if type(image_tensor) is list:
                image_tensor = [
                    _image.to(dtype=torch.float16, device=self.device)
                    for _image in image_tensor
                ]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        elif task_type == "video":  # For video task
            image_tensor = []
            max_frames = gen_kwargs.get("sample_frames", self.max_frames_num)
            if "sample_frames" in gen_kwargs:
                gen_kwargs.pop("sample_frames")

            try:
                if self.video_decode_backend == "decord":
                    frames = self.load_video(visuals, max_frames)

                frames = (
                    self._image_processor.preprocess(frames, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .cuda()
                )
                image_tensor.append(frames)
            except Exception as e:
                eval_logger.error(f"Error {e} in loading video")
                image_tensor = None

            task_type = "video"

        if (
            image_tensor is not None
            and len(image_tensor) != 0
            and DEFAULT_IMAGE_TOKEN not in context
        ):
            """
            Three senarios:
            1. No image, and there for, no image token should be added.
            2. image token is already specified in the context, so we don't need to add it.
            3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
            4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
            """
            if task_type == "image":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(visuals)
                    if isinstance(visuals, list)
                    else [DEFAULT_IMAGE_TOKEN]
                )
            elif task_type == "video":
                image_tokens = (
                    [DEFAULT_IMAGE_TOKEN] * len(frames)
                    if self.token_strategy == "multiple"
                    else [DEFAULT_IMAGE_TOKEN]
                )

            image_tokens = " ".join(image_tokens)
            question = image_tokens + "\n" + context
        else:
            question = context

        # This is much safer for llama3, as we now have some object type in it
        if "llama_3" in self.conv_template:
            conv = copy.deepcopy(conv_templates[self.conv_template])
        else:
            conv = conv_templates[self.conv_template].copy()

        for prev_conv in requests["prev_conv"]:
            conv.append_message(conv.roles[0], prev_conv[0])
            conv.append_message(conv.roles[1], prev_conv[1])

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt_question = conv.get_prompt()
        question_input.append(prompt_question)

        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        input_ids_list = [
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            for prompt in question_input
        ]
        pad_token_ids = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        input_ids = self.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_ids
        ).to(self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )
        if task_type == "image":
            gen_kwargs["image_sizes"] = [
                visual.size for visual in visuals
            ]  # (width, height)
        elif task_type == "video":
            gen_kwargs["modalities"] = ["video"]
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: attention to this major generation step...
        if "image_aspect_ratio" in gen_kwargs.keys():
            gen_kwargs.pop("image_aspect_ratio")

        max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        num_image_tokens = (
            question.count(DEFAULT_IMAGE_TOKEN)
            * self.model.get_vision_tower().num_patches
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
        )

        gen_kwargs["max_new_tokens"] = min(
            gen_kwargs["max_new_tokens"],
            max_context_length - input_ids.shape[-1] - num_image_tokens,
        )

        if gen_kwargs["max_new_tokens"] < 1:
            yield json.dumps(
                {
                    "text": question
                    + "Exceeds max token length. Please start a new conversation, thanks.",
                    "error_code": 0,
                }
            ).encode() + b"\0"
            return

        print(f"gen_kwargs: {gen_kwargs}")
        try:
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    inputs=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    use_cache=self.use_cache,
                    streamer=streamer,
                    **gen_kwargs,
                ),
            )
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[: -len(stop_str)]
                yield json.dumps(
                    {"text": generated_text, "error_code": 0}
                ).encode() + b"\0"
            # with torch.inference_mode():
            #     cont = self.model.generate(
            #         input_ids,
            #         attention_mask=attention_masks,
            #         pad_token_id=pad_token_ids,
            #         images=image_tensor,
            #         use_cache=self.use_cache,
            #         **gen_kwargs,
            #     )

            # text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        except Exception as e:
            raise e

def image_demo(model):
# Get the directory of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the visual file relative to the current script file
    visual_path = os.path.join(current_dir, "assets", "lmms-eval.png")
    image = Image.open(visual_path).convert("RGB")
    
    input_visuals = [image]
    input_context = "What is shown in this image?"
    task_type = "image"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    try:
        prev = 0
        for x in model.stream_generate_until(query, gen_kwargs):
            output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)

        print("\n")
    except Exception as e:
        print(e)

def video_demo(model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visual_path = os.path.join(current_dir, "assets", "water.mp4")
    input_visuals = [visual_path]
    input_context = "What is shown in this video?"
    task_type = "video"
    gen_kwargs = {"max_new_tokens": 1024, "temperature": 0, "do_sample": False, "sample_frames": 128}
    query = {
        "visuals": input_visuals,
        "context": input_context,
        "task_type": task_type,
        "prev_conv": [],
    }
    try:
        prev = 0
        for x in model.stream_generate_until(query, gen_kwargs):
            output = json.loads(x.decode("utf-8").strip("\0"))["text"].strip()
            print(output[prev:], end="", flush=True)
            prev = len(output)

        print("\n")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    model = LongVA(pretrained="lmms-lab/LongVA-7B-DPO", model_name="llava_qwen")
    image_demo(model)
    video_demo(model)