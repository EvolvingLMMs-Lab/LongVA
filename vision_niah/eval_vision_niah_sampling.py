from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from datasets import load_dataset
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)

model_path = "lmms-lab/LongVA-7B"
video_path = "vision_niah/data/haystack_videos/movie.mp4"
haystack_frames = 256 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": False, "use_cache": True, "max_new_tokens": 1024}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="auto") 

preprompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
postprompt = "<|im_end|>\n<|im_start|>assistant\n"
#video input


vniah_needle_dataset = load_dataset("lmms-lab/v_niah_needles")["test"]
question = vniah_needle_dataset[1]["question"]
image = vniah_needle_dataset[1]["image"].convert("RGB")
answer = vniah_needle_dataset[1]["answer"]
images_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)



prompt = preprompt + "<image>" + question + postprompt
print(prompt)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, haystack_frames, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
# insert image_tensor in the middle
video_tensor = torch.cat([video_tensor[:len(video_tensor)//2], images_tensor, video_tensor[len(video_tensor)//2:]], dim=0)
# insert at the very end
# video_tensor = torch.cat([video_tensor, images_tensor], dim=0)
with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("Output:", outputs)
print("Answer:", answer)