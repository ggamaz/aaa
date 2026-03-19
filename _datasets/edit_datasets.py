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
import random
import torch
try:
    from aoss_client.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        Client = None
from glob import glob
from .utils import crop2square, resize_image_fix_pixels, resize_image_dynamic
from einops import rearrange
import numpy as np


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
            # import pdb; pdb.set_trace()
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



class MultiImageEditDataset(CaptionDataset):
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
                source_images = []
                for img_path in data_sample['input_image']:
                    img = Image.open(os.path.join(self.image_folder, img_path)).convert('RGB')
                    source_images.append(img)
                target_image = Image.open(os.path.join(self.image_folder, data_sample['output_image'])).convert('RGB')
            else:
                source_images =[]
                for img_path in data_sample['input_image']:
                    img = Image.open(img_path).convert('RGB')
                    source_images.append(img)
                target_image = Image.open(data_sample['output_image']).convert('RGB')

            prompt = data_sample['instruction']

            pixel_values_src = []
            for img in source_images:
                pixel_values_src.append(self._process_image(img))
            pixel_values = self._process_image(target_image)

            data = self._process_text(prompt) if self.tokenizer is not None else dict()

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder,type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class ReconstructDataset(CaptionDataset):
    def _process_image(self, image):
        assert self.image_process != 'crop2square'
        return super()._process_image(image)['pixel_values']

    def __getitem__(self, idx):
        if self.debug:
            idx = 0
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            prompt = "Keep the image as it is."
            pixel_values = pixel_values_src = self._process_image(image)

            data = self._process_text(prompt) if self.tokenizer is not None else dict()

            data.update(
                pixel_values_src=pixel_values_src, pixel_values=pixel_values,
                image_dir=self.image_folder, image_file=data_sample['image'],
                type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


if __name__ == "__main__":
    import traceback
    from transformers import AutoTokenizer

    QWEN_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    PROMPT_TEMPLATE = dict(
        IMG_START_TOKEN='<|vision_start|>',
        IMG_END_TOKEN='<|vision_end|>',
        IMG_CONTEXT_TOKEN='<|image_pad|>',
        IMG_START_TOKEN_FOR_GENERATION=False,
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
        GENERATION='Generate an image: {input}',
        CFG='Generate an image.',
    )

    # ── 修改这两个路径 ────────────────────────────────────────────────
    DATA_PATH    = "/path/to/demo.json"
    IMAGE_FOLDER = "/path/to/images"       # 若路径已在 json 里写绝对路径则设 None
    # ─────────────────────────────────────────────────────────────────

    print("=" * 60)
    print("Step 1: 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH, trust_remote_code=True, padding_side='left')
    print(f"  vocab size: {tokenizer.vocab_size}")

    print("\nStep 2: 查看 JSON 原始第一条，确认字段名")
    import json
    with open(DATA_PATH) as f:
        raw = json.load(f)
    print(f"  共 {len(raw)} 条样本")
    print(f"  第一条字段: {list(raw[0].keys())}")
    print(f"  第一条内容: {raw[0]}")

    print("\nStep 3: 构建 MultiImageEditDataset")
    dataset = MultiImageEditDataset(
        data_path=DATA_PATH,
        image_folder=IMAGE_FOLDER,
        tokenizer=tokenizer,
        prompt_template=PROMPT_TEMPLATE,
        image_size=512,
        image_length=256,
        image_process='dynamic',
        unit_image_size=32,
        max_length=1024,
    )
    print(f"  dataset size: {len(dataset)}")

    print("\nStep 4: 逐条取前 3 条，打印详情")
    for i in range(min(3, len(dataset))):
        print(f"\n  --- sample {i} ---")
        try:
            sample = dataset._get_item(i)      # 直接调，不走 retry，报错立即可见
            print(f"  keys            : {list(sample.keys())}")
            print(f"  text            : {sample['text'][:80]}")
            print(f"  pixel_values    : {sample['pixel_values'].shape}  "
                  f"range [{sample['pixel_values'].min():.2f}, {sample['pixel_values'].max():.2f}]")
            print(f"  pixel_values_src: {len(sample['pixel_values_src'])} refs")
            for j, src in enumerate(sample['pixel_values_src']):
                print(f"    ref[{j}]: {src.shape}  range [{src.min():.2f}, {src.max():.2f}]")
            if 'input_ids' in sample:
                print(f"  input_ids       : {sample['input_ids'].shape}")
                print(f"  decoded prompt  : {tokenizer.decode(sample['input_ids'][:60])}...")
        except Exception:
            print(f"  [ERROR] sample {i} 报错如下：")
            traceback.print_exc()

    print("\nStep 5: 测试 DataLoader collate")
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return dict(
            pixel_values_src=[b['pixel_values_src'] for b in batch],
            pixel_values    =[b['pixel_values']      for b in batch],
            texts           =[b['text']              for b in batch],
        )

    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    try:
        batch = next(iter(loader))
        print(f"  batch keys           : {list(batch.keys())}")
        print(f"  batch texts          : {[t[:40] for t in batch['texts']]}")
        print(f"  batch pixel_values[0]: {batch['pixel_values'][0].shape}")
        print(f"  batch pixel_values_src[0] refs: {len(batch['pixel_values_src'][0])}")
        print("\n[PASS] DataLoader 正常")
    except Exception:
        print("[FAIL] DataLoader 报错：")
        traceback.print_exc()