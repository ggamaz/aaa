import math

import torch
import numpy as np
from einops import rearrange
from PIL import Image
import os

from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import torch

Client = None
from glob import glob
from .utils import crop2square, resize_image_fix_pixels, resize_image_dynamic, paired_random_crop
from einops import rearrange
import numpy as np
import random
import torchvision.transforms.functional as TF

class CaptionDataset(Dataset):
    def __init__(self,
                 data_path,
                 image_folder=None,
                 debug=False,
                 image_processor=None,
                 image_process='crop2square',
                 ceph_folder=None,
                 latents_ceph_folder=None,
                 ceph_config=None,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=2048,
                 min_image_size=80,
                 image_size=256,
                 image_length=256,
                 unit_image_size=32,
                 image_tokens_folder=None,
                 image_latents_folder=None,
                 cap_folder=None,
                 cap_source='caption',
                 tokenizer_kwargs=dict(add_special_tokens=True),
                 unconditional=0.1
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_folder = image_folder
        self.cap_folder = cap_folder
        self.cap_source = cap_source
        self.debug = debug

        self.image_processor = image_processor  # 直接使用，调用方负责实例化
        self.tokenizer = tokenizer              # 直接使用，调用方负责实例化
        self.prompt_template = prompt_template

        self.max_length = max_length
        self.image_process = image_process
        self.image_length = image_length
        self.image_tokens_folder = image_tokens_folder
        self.image_latents_folder = image_latents_folder
        self.min_image_size = min_image_size
        self.image_size = image_size
        self.unit_image_size = unit_image_size
        self.unconditional = unconditional
        self.tokenizer_kwargs = tokenizer_kwargs

        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder
        self.ceph_config = ceph_config
        self.latents_ceph_folder = latents_ceph_folder
        self.use_ceph = ((Client is not None) and (ceph_config is not None) and os.path.exists(ceph_config))

    def _load_data(self, data_path: str):      # image path and annotation path are saved in a json file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            json_files = glob(f"{data_path}/*.json")
            data_list = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data_list += json.load(f)

            self.data_list = data_list

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_ceph(self, ceph_path):
        if self.FILE_CLIENT is None:
            self.FILE_CLIENT = Client(self.ceph_config)
        data_bytes = self.FILE_CLIENT.get(ceph_path)

        return io.BytesIO(data_bytes)

    def _read_image(self, image_file):
        if self.image_folder is None:
            assert self.use_ceph
            assert self.ceph_folder is not None
            image = Image.open(
                self._read_ceph(
                    os.path.join(self.ceph_folder, image_file)
                )
            )
        else:
            image = Image.open(
                os.path.join(self.image_folder, image_file)
            )
        assert image.width > self.min_image_size and image.height > self.min_image_size, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _read_json(self, annotation_file):
        if self.cap_folder is None:
            assert self.use_ceph
            assert self.ceph_folder is not None
            annotation = json.load(
                self._read_ceph(
                    os.path.join(self.ceph_folder, annotation_file)
                )
            )
        else:
            with open(os.path.join(self.cap_folder, annotation_file), 'r') as f:
                annotation = json.load(f)

        return annotation

    def _process_image(self, image):
        data = dict()
        if self.image_process == 'crop2square':
            image = crop2square(image)
            image = image.resize(size=(self.image_size, self.image_size))
        elif self.image_process == 'dynamic':   # dynamic and make sure the largest edge <= self.image_size
            image = resize_image_dynamic(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'fix_pixels': # fix pixels contain radio of image
            image = resize_image_fix_pixels(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'resize2square':
            image = image.resize(size=(self.image_size, self.image_size))
        else:
            raise NotImplementedError

        # assert image.width <= self.image_size
        # assert image.height <= self.image_size
        assert image.width % self.unit_image_size == 0
        assert image.height % self.unit_image_size == 0

        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)
        return data

    def _process_text(self, text):
        if self.tokenizer is None:
            return {}
        if random.uniform(0, 1) < self.unconditional:
            prompt = self.prompt_template['CFG']
        else:
            prompt = self.prompt_template['GENERATION'].format(input=text.strip())

        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        if self.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
            prompt += self.prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]

            if self.image_tokens_folder is not None:
                image_tokens = torch.load(os.path.join(self.image_tokens_folder,
                                                       data_sample['image'] + '.pt')).long()
                data = dict(image_tokens=image_tokens)
            elif self.latents_ceph_folder is not None:
                image_latents = torch.load(
                    self._read_ceph(
                        os.path.join(
                            self.latents_ceph_folder, data_sample['image'] + '.pt'
                        )
                    )
                )
                data = dict(image_latents=image_latents)
            elif self.image_latents_folder is not None:
                image_latents = torch.load(os.path.join(self.image_latents_folder,
                                                        data_sample['image'] + '.pt'))
                data = dict(image_latents=image_latents)
            else:
                image = self._read_image(data_sample['image']).convert('RGB')
                data = self._process_image(image)

            caption = self._read_json(data_sample['annotation'])[self.cap_source]
            # caption = self._read_json(data_sample['annotation'])
            # print(caption)

            data.update(self._process_text(caption))
            data['pixel_init'] = image
            data.update(image_dir=self.image_folder, image_file=data_sample['image'],
                        type='text2image',text=caption)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()



class ImageEditDataset(CaptionDataset):
    def _process_image(self, image):
        assert self.image_process != 'crop2square'
        return super()._process_image(image)['pixel_values']
        # image = image.resize(size=(self.image_size, self.image_size))
        # pixel_values = torch.from_numpy(np.array(image)).float()
        # pixel_values = pixel_values / 255
        # pixel_values = 2 * pixel_values - 1
        # pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        # return pixel_values

    def _process_text(self, text):
        prompt_template = self.prompt_template
        image_tokens = prompt_template['IMG_START_TOKEN'] + \
                       prompt_template['IMG_CONTEXT_TOKEN'] * self.image_length + \
                       prompt_template['IMG_END_TOKEN']
        prompt = f'{image_tokens}\n{text}'
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        if self.prompt_template.get('IMG_START_TOKEN_FOR_GENERATION', True):
            prompt += prompt_template['IMG_START_TOKEN']
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', **self.tokenizer_kwargs)[0]

        return dict(input_ids=input_ids)

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            if self.image_folder is not None:
                source_image = Image.open(os.path.join(self.image_folder,data_sample['input_image'][0])).convert('RGB')
                target_image = Image.open(os.path.join(self.image_folder,data_sample['output_image'])).convert('RGB')
            else:
                source_image = Image.open(data_sample['input_image'][0]).convert('RGB')
                target_image = Image.open(data_sample['output_image']).convert('RGB')
            # prompt = self._read_json(data_sample['annotation'])[self.cap_source]
            prompt = data_sample['instruction']

            pixel_values_src = self._process_image(source_image)
            pixel_values = self._process_image(target_image)

            data = self._process_text(prompt) if self.tokenizer is not None else dict()

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder,type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()

class TwoImageEditDataset(ImageEditDataset):
    
    def _process_image_group(self, images):
        w, h = images[0].size
        if w < self.image_size or h < self.image_size:
            ratio = max(self.image_size / w, self.image_size / h)
            new_w, new_h = math.ceil(w * ratio), math.ceil(h * ratio)
            images = [img.resize((new_w, new_h), Image.BILINEAR) for img in images]

        cropped_images = paired_random_crop(images, size=self.image_size)

        cw, ch = cropped_images[0].size
        assert cw % self.unit_image_size == 0, f"Crop width {cw} is not divisible by {self.unit_image_size}"
        assert ch % self.unit_image_size == 0, f"Crop height {ch} is not divisible by {self.unit_image_size}"

        pixel_values = torch.from_numpy(np.array(cropped_images)).float()
        pixel_values = pixel_values / 255.0
        pixel_values = 2.0 * pixel_values - 1.0
        pixel_values = rearrange(pixel_values, 'b h w c -> b c h w').contiguous()

        return pixel_values[0], pixel_values[1], pixel_values[2]

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            if self.image_folder is None:
                self.image_folder = ""    
                
            visible_image = Image.open(os.path.join(self.image_folder, data_sample['input_v_image'])).convert('RGB')
            infrared_image = Image.open(os.path.join(self.image_folder, data_sample['input_ir_image'])).convert('RGB')
            target_image = Image.open(os.path.join(self.image_folder, data_sample['output_image'])).convert('RGB')
            visible_pixel_values, infrared_pixel_values, target_pixel_values = self._process_image_group([visible_image, infrared_image, target_image])

            # 随机选择 Prompt 并打上类型标签
            prompt_visual = data_sample.get('instruction', 'fuse for visual quality')
            prompt_downstream = data_sample.get('instruction_downstream', 'fuse for downstream detection task')
            
            return dict(
                pixel_values_src=[visible_pixel_values, infrared_pixel_values],
                pixel_values=target_pixel_values,
                texts=[prompt_visual, prompt_downstream],
                prompt_types=['visual', 'downstream']
            )
            
        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()

class MaskImageEditDataset(ImageEditDataset):
    
    def _process_image_group(self, images):
        w, h = images[0].size
        if w < self.image_size or h < self.image_size:
            ratio = max(self.image_size / w, self.image_size / h)
            new_w, new_h = math.ceil(w * ratio), math.ceil(h * ratio)
            images = [img.resize((new_w, new_h), Image.BILINEAR) for img in images]

        max_retries = 5
        
        for _ in range(max_retries):
            cropped_images = paired_random_crop(images, size=self.image_size)
            mask_crop = cropped_images[3] 
            
            tiny_mask = mask_crop.resize((64, 64), Image.NEAREST)
            mask_np = np.array(tiny_mask.convert('L'))
            
            fg_pixels = np.sum(mask_np > 127)
            total_pixels = 64 * 64
            fg_ratio = fg_pixels / total_pixels
            
            if fg_ratio > 0.01:
                break 
        pixel_values = torch.stack([TF.pil_to_tensor(img) for img in cropped_images]) 
        pixel_values = pixel_values.float() / 255.0 * 2.0 - 1.0  # 归一化到 [-1, 1]，在 GPU 上执行
        return pixel_values[0], pixel_values[1], pixel_values[2], pixel_values[3]

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            if self.image_folder is None:
                self.image_folder = ""    
                
            # 1. 加载输入图像
            visible_image = Image.open(os.path.join(self.image_folder, data_sample['input_v_image'])).convert('RGB')
            infrared_image = Image.open(os.path.join(self.image_folder, data_sample['input_ir_image'])).convert('RGB')
            
            # 2. 加载视觉融合的目标图像
            target_image = Image.open(os.path.join(self.image_folder, data_sample['output_image'])).convert('RGB')
            
            # 3. [新增] 加载下游分割任务的 Mask 图像
            # 注意：即使 Mask 是单通道灰度图，也建议 .convert('RGB')。
            # 因为 VAE 默认需要 3 通道输入，转成 RGB 后 [0, 0, 0] 和 [255, 255, 255] 依然是完美的二值边界。
            mask_image = Image.open(os.path.join(self.image_folder, data_sample['output_mask'])).convert('RGB')

            # 统一送入处理函数，保持空间对齐
            visible_pv, infrared_pv, target_pv, mask_pv = self._process_image_group(
                [visible_image, infrared_image, target_image, mask_image]
            )

            # 4. 读取我们在 JSON 中更新过的 Prompt 键名
            prompt_visual = data_sample.get('instruction_fusion', 'Fuse the images to restore high quality details.')
            prompt_segmentation = data_sample.get('instruction_segmentation', 'Segment the target object and output the mask.')
            
            # 5. 构建返回字典
            return dict(
                pixel_values_src=[visible_pv, infrared_pv],
                pixel_values=target_pv,            # 视觉任务的目标图
                pixel_masks=mask_pv,               # [新增] 分割任务的目标 Mask 图
                texts=[prompt_visual, prompt_segmentation],
                prompt_types=['visual', 'segmentation']  # [修改] 标签名对齐我们 Loss 函数里的逻辑
            )
            
        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()