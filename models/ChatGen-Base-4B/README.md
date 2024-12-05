---
license: apache-2.0
base_model:
- OpenGVLab/InternVL2-4B
pipeline_tag: image-text-to-text
library_name: transformers
---

# ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting

<div align="center">

[\[🏠Homepage\]](https://chengyou-jia.github.io/ChatGen-Home/) [\[💻Code\]](https://github.com/chengyou-jia/ChatGen) [\[🚀Quick Start\]](#quick-start) [\[📝Paper\]](https://arxiv.org/abs/2411.17176) [\[🤗Models\]](https://huggingface.co/ChengyouJia/ChatGen-Base-4B)[\[🤗Data\]](https://huggingface.co/datasets/ChengyouJia/ChatGenBench)

</div>

## Overview
![ChatGen](./case_step.png)

ChatGen aims to automate tedious steps in text-to-image, allowing users to simply describe their needs in a freestyle chatting way.



## ChatGen-Base-4B 

`ChatGen-Base-4B` is a MLLM finetuned from InternVL-4B. By taking as input a system prompt, and freestyle user query, 
the model generates suitable prompts, appropriate models, and specific arguments.


### Installation
To use `ChatGen-Base-4B`, first install the necessary dependencies:
```bash
pip install transformers
```

### Example Inference Code
Inference code example:
```python
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'ChengyouJia/ChatGen-Base-4B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

sys_singlemodal = """
You are a user requirements translation expert. I have a freestyle prompt written by a non professional user for text-to-image tasks. Please convert the content of this freestyle prompt into professional prompt and professional negativePrompt, and provide the model and its parameters that are most suitable for the user's text-to-image task.
Here is the content I need you to convert:
"""

sys_multimodal = """
You are a user requirements translation expert. I have a freestyle prompt written by a non professional user for text-to-image tasks.
Additionally, a general user provide several reference images, indicating that they want the final generated image to have a style similar to those images. You should combine the reference images to convert the content of the freestyle prompt into professional prompt and professional negativePrompt, and provide the model and its parameters that are most suitable for the user's text-to-image task.
Here are the reference images and content I need you to convert:
"""

# set the max number of tiles in `max_num`
pixel_values = None
<!-- pixel_values = load_image(<image_path>, max_num=6).to(torch.bfloat16).cuda() -->
generation_config = dict(max_new_tokens=1024, do_sample=True)

question = "Whip up a cool sci-fi robot girl, colorful and detailed from waist up, y'know?"

input = sys_singlemodal + question 
response, history = model.chat(tokenizer, None, input, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```
```

## Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@article{jia2024chatgen, 
  title={ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting}, 
  author={Jia, Chengyou and Xia, Changliang and Dang, Zhuohang and Wu, Weijia and Qian, Hangwei and Luo, Minnan}, 
  journal={arXiv preprint arXiv:2411.17176}, 
  year={2024}
}
```
