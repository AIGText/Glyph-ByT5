from curses import meta
from dataclasses import dataclass
from PIL import ImageDraw, ImageFont, Image
import random
import json
import webcolors
import copy
import time
import torch
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from dataclasses import dataclass

FONT_FILE_PATH = 'assets/fonts'
CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
        'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '!', '@', '#', '$', '%', '^', '&', '*',
        '(', ')', '-', '_', '=', '+', '[', ']', '{', '}']

def is_color_close_to_black(hex_color, threshold=50):  
    r = int(hex_color[1:3], 16)  
    g = int(hex_color[3:5], 16)  
    b = int(hex_color[5:7], 16)  
    distance = (r**2 + g**2 + b**2)**0.5  
    return distance < threshold  

def generate_random_color(threshold=50):  
    while True:  
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  
        if not is_color_close_to_black(color, threshold):  
            return color  

def get_color(hex_color, threshold=50):  
    if hex_color is None or is_color_close_to_black(hex_color, threshold):  
        return generate_random_color(threshold=threshold)  
    else:  
        return hex_color

@dataclass
class TextInstance:
    _font_mapping = None

    text: str
    width: float
    height: float
    left: float
    top: float
    angle: float
    font: str
    color: str = None

    def as_tuple(self):
        return (self.text, (self.left, self.top, self.width, self.height, self.angle))
    
    def get_font_file(self):
        return f"{FONT_FILE_PATH}/{self.font}.ttf"

def closest_color(requested_color):  
    min_colors = {}  
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():  
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)  
        rd = (r_c - requested_color[0]) ** 2  
        gd = (g_c - requested_color[1]) ** 2  
        bd = (b_c - requested_color[2]) ** 2  
        min_colors[(rd + gd + bd)] = name  
    return min_colors[min(min_colors.keys())]

def convert_rgb_to_names(rgb_tuple):  
    try:  
        color_name = webcolors.rgb_to_name(rgb_tuple)  
    except ValueError:  
        color_name = closest_color(rgb_tuple)  
    return color_name

def convert_name_to_hex(color_name):
    hex_value = webcolors.name_to_hex(color_name)
    return hex_value

def get_caption_text(texts, styles, font_idx_dict=None, color_idx_dict=None):
    prompt = ""
    '''
    Text "{text}" in {color}, {type}.
    '''
    for text, style in zip(texts, styles):
        text_prompt = f'Text "{text}"'

        attr_list = []
        if 'color' in style:
            hex_color = style["color"]  
            rgb_color = webcolors.hex_to_rgb(hex_color)
            # get color name  
            color_name = convert_rgb_to_names(rgb_color)
            if color_idx_dict is not None:
                attr_list.append(f"<color-{color_idx_dict[color_name]}>")
            else:
                attr_list.append(color_name)
        if 'font-family' in style:
            font_name = style["font-family"]
            if font_idx_dict is not None and font_name in font_idx_dict:
                attr_list.append(f"<font-{font_idx_dict[font_name]}>")
            else:
                attr_list.append(font_name)

        if len(attr_list) > 0:
            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
        text_prompt += ". "

        prompt = prompt + text_prompt
    return prompt

def get_caption_text_wo_style(texts):
    prompt = ""
    '''
    Text "{text}" in {color}, {type}.
    '''
    for text in texts:
        text_prompt = f'Text "{text}"'
        text_prompt += ". "
        prompt = prompt + text_prompt
    return prompt

def replace_char_strategy(text):
    replaceable_idx = [i for char, i in zip(text, range(len(text))) if char != ' ' and char != '\n']
    if len(replaceable_idx) == 0:
        return text
    replace_idx = random.choice(replaceable_idx)
    new_char = text[replace_idx]
    while new_char == text[replace_idx]:
        new_char = random.choice(CHARS)
    text = text[:replace_idx] + new_char + text[replace_idx + 1:]
    return text

def repeat_char_strategy(text):
    words = text.split(' ')
    repeatable_idx = [i for word, i in zip(words, range(len(words))) if word.replace('\n', '') != '']
    if len(repeatable_idx) == 0:
        return text
    repeat_idx = random.choice(repeatable_idx)
    word = words[repeat_idx]
    repeat_char_idx = random.randint(0, len(word) - 1)
    while word[repeat_char_idx] == '\n':
        repeat_char_idx = random.randint(0, len(word) - 1)
    repeat_times = random.randint(2, 6)
    new_word = word[:repeat_char_idx] + word[repeat_char_idx] * repeat_times + word[repeat_char_idx + 1:]
    words[repeat_idx] = new_word
    text = ' '.join(words)
    return text

def add_char_strategy(text):
    words = text.split(' ')
    addable_idx = [i for word, i in zip(words, range(len(words))) if word.replace('\n', '') != '']
    if len(addable_idx) == 0:
        return text
    add_idx = random.choice(addable_idx)
    word = words[add_idx]
    # len(word) for inserting at the end of the word
    insert_char_idx = random.randint(0, len(word))
    add_times = random.randint(1, 6)
    append_str = ''
    for _ in range(add_times):
        append_str += random.choice(CHARS)
    new_word = word[:insert_char_idx] + append_str + word[insert_char_idx:]
    words[add_idx] = new_word
    text = ' '.join(words)
    return text

def drop_char_strategy(text):
    dropable_idx = [i for char, i in zip(text, range(len(text))) if char != ' ' and char != '\n']
    drop_idx = random.choice(dropable_idx)
    text = text[:drop_idx] + text[drop_idx + 1:]
    return text

def wrong_word_strategy(text):
    modify_rate = random.random()
    words = text.split(' ')
    modifiable_idx = [i for word, i in zip(words, range(len(words))) if (('\n' not in word) or (word[-1] == '\n' and '\n' not in word[:-1])) and len(word) > 0]
    if len(modifiable_idx) == 0:
        return text
    modify_idx = random.choice(modifiable_idx)
    word = words[modify_idx]
    post_fix = '\n' if word[-1] == '\n' else ''
    word = word.replace('\n', '')
    num_char_change = max(1, int(len(word) * modify_rate))
    for _ in range(num_char_change):
        idx = random.randint(0, len(word) - 1)
        strategy = random.randint(0, 3)
        if strategy == 0:
            # repeat char
            repeat_times = random.randint(2, 6)
            word = word[:idx] + word[idx] * repeat_times + word[idx + 1:] 
        elif strategy == 1:
            # add char
            add_times = random.randint(1, 6)
            append_str = ''
            for _ in range(add_times):
                append_str += random.choice(CHARS)
            word[:idx] + append_str + word[idx:]
        elif strategy == 2:
            # replace char
            new_char = word[idx]
            while new_char == word[idx]:
                new_char = random.choice(CHARS)
            word = word[:idx] + new_char + word[idx + 1:]
        elif strategy == 3:
            # drop char
            word = word[:idx] + word[idx + 1:]
    word = word + post_fix
    words[modify_idx] = word
    text = ' '.join(words)
    return text

def repeat_word_strategy(text):
    words = text.split(' ')
    repeatable_idx = [i for word, i in zip(words, range(len(words))) if word.replace('\n', '') != '']
    if len(repeatable_idx) == 0:
        return text
    repeat_idx = random.choice(repeatable_idx)
    word = words[repeat_idx]
    repeat_part = word.split('\n')[0]
    repeat_times = random.randint(2, 4)
    new_word = (repeat_part + ' ') * repeat_times + repeat_part
    words[repeat_idx] = new_word
    text = ' '.join(words)
    return text

def drop_word_strategy(text):
    words = text.split(' ')
    dropable_idx = [i for word, i in zip(words, range(len(words))) if word.replace('\n', '') != '']
    if len(dropable_idx) == 0:
        return text
    drop_times = min(random.randint(1, 4), len(dropable_idx))
    drop_idxes = random.sample(dropable_idx, drop_times)
    drop_idxes = sorted(drop_idxes, reverse=True)
    for drop_idx in drop_idxes:
        word = words[drop_idx]
        if '\n' in word:
            new_word = word[word.find('\n'):]
            words[drop_idx] = new_word
        else:
            words.pop(drop_idx)
    text = ' '.join(words)
    return text

def get_multiline_text_autowrap(text, font, bbox):
    max_width = bbox[2]
    lines = []
    words = text.split()
    while words:
        line = ''  
        line += (words.pop(0) + ' ')
        while (words and font.getlength(line + words[0]) <= max_width):  
            line += (words.pop(0) + ' ')
        lines.append(line.strip())  
    return "\n".join(lines)


class GlyphBoxDataset(Dataset):
    AUG_STRATEGIES = [
        replace_char_strategy,
        repeat_char_strategy,
        add_char_strategy,
        drop_char_strategy,
        wrong_word_strategy,
        repeat_word_strategy,
        drop_word_strategy,
    ]
    def __init__(
        self, 
        num_negative,
        transform,
        max_tries=5,
        meta_path=None,
        font_ann_path=None,
        color_ann_path=None,
        font_special_token=False,
        color_special_token=False,
        train_color_prob=0,
        train_font_prob=0,
        auto_wrap=False,
        max_box_per_im=None,
        patch_size=14,
    ):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.num_negative = num_negative
        self.transform = transform
        self.max_tries = max_tries
        self.meta = meta
        with open(font_ann_path, 'r') as f:
            self.font_idx_dict = json.load(f)
            self.font_idx_list = list(self.font_idx_dict.items())
        with open(color_ann_path, 'r') as f:
            self.color_idx_dict = json.load(f)
            self.color_idx_list = list(self.color_idx_dict.items())
        self.font_special_token = font_special_token
        self.color_special_token = color_special_token
        self.train_color_prob = train_color_prob
        self.train_font_prob = train_font_prob
        self.auto_wrap = auto_wrap
        self.max_box_per_im = max_box_per_im

        self.img_h = 224 // patch_size
        self.img_w = 224 // patch_size
        self.text_len = 512
        self.patch_size = patch_size

    @staticmethod
    def render_text(image: Image, text_instance: TextInstance, resolution: int = 224, auto_wrap: bool = False):
        text, bbox = text_instance.as_tuple()
        bbox = list(map(lambda x: x * resolution / 1645, bbox))
        font_file = text_instance.get_font_file()

        # Binary search for the largest font size that fits the bbox
        font_size = 1
        font_size_upper_bound = 200
        font_size_lower_bound = 1
        draw = ImageDraw.Draw(image)
        font_x, font_y = 0, 0
        while font_size_lower_bound < font_size_upper_bound:
            try:
                font = ImageFont.truetype(font_file, size=font_size)
            except:
                font_list = font_file.split('-')
                prefix = "".join(font_list[:-1])
                font_file = f"{prefix}-0.ttf"
                font = ImageFont.truetype(font_file, size=font_size)
            try:
                if auto_wrap:
                    text = get_multiline_text_autowrap(text, font, bbox)
                    x_offset, y_offset = font.getbbox(text)[:2]
                    x, y = bbox[0] - x_offset, bbox[1] - y_offset
                    left, top, right, bottom = draw.multiline_textbbox((x, y), text, font=font, align="center")
                else:
                    x_offset, y_offset = font.getbbox(text)[:2]
                    x, y = bbox[0] - x_offset, bbox[1] - y_offset
                    left, top, right, bottom = draw.multiline_textbbox((x, y), text, font=font, align="center")
            except:
                return font_size
            
            width = right - left
            height = bottom - top

            if width > bbox[2] or height > bbox[3]:
                font_size_upper_bound = font_size - 1
            else:
                font_size_lower_bound = font_size
                font_x, font_y = width, height
            font_size = (font_size_lower_bound + font_size_upper_bound + 1) // 2

        try:
            font = ImageFont.truetype(font_file, size=font_size)
        except:
            font_list = font_file.split('-')
            prefix = "".join(font_list[:-1])
            font_file = f"{prefix}-0.ttf"
            font = ImageFont.truetype(font_file, size=font_size)

        fill_color = text_instance.color
        x_offset, y_offset = font.getbbox(text)[:2]
        x, y = bbox[0] - x_offset, bbox[1] - y_offset
        if auto_wrap:
            text = get_multiline_text_autowrap(text, font, bbox)
        draw.multiline_text(
            (x + (bbox[2] - font_x) // 2, y + (bbox[3] - font_y) // 2),
            text,
            font=font,
            fill=fill_color,
            align="center",
        )
        return font_size
    
    def clamp_len(self, l, max_len):
        return max(0, min(l, max_len))

    def check_box(self, bbox, canvas_len=1645):
        bbox[0] = self.clamp_len(bbox[0], canvas_len)
        bbox[1] = self.clamp_len(bbox[1], canvas_len)
        bbox[2] = self.clamp_len(bbox[2], canvas_len - bbox[0])
        bbox[3] = self.clamp_len(bbox[3], canvas_len - bbox[1])
        return bbox

    def get_caption_and_img(self, texts, styles, bboxes, auto_wrap=False):
        font_dict = self.font_idx_dict if self.train_font_prob > 0 else None
        color_dict = self.color_idx_dict if self.train_color_prob > 0 else None
        caption = get_caption_text(texts, styles, font_idx_dict=font_dict, color_idx_dict=color_dict)
        img = Image.new("RGB", (224, 224), (0, 0, 0))
        font_sizes = []
        for text, style, bbox in zip(texts, styles, bboxes):
            bbox = self.check_box(bbox)
            text_instance = TextInstance(
                text = text,
                left = bbox[0],
                top = bbox[1],
                width = bbox[2],
                height = bbox[3],
                angle = bbox[4],
                font = style.get("font-family"),
                color = style.get("color", None)
            )
            font_size = self.render_text(img, text_instance, auto_wrap=auto_wrap)
            font_sizes.append(font_size)
        return caption, img, font_sizes

    def nms(self, texts, styles, bboxes):
        np_bboxes = np.array(bboxes)
        bboxes_area = np_bboxes[:, 2] * np_bboxes[:, 3]
        index = bboxes_area.argsort()[::-1]

        x1 = np_bboxes[:, 0]
        y1 = np_bboxes[:, 1]
        x2 = np_bboxes[:, 2] + np_bboxes[:, 0]
        y2 = np_bboxes[:, 3] + np_bboxes[:, 1]

        result = []
        while index.size > 0:
            i = index[0]
            result.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)
            overlaps = w * h

            idx = np.where(overlaps <= 0)[0]
            index = index[idx + 1]

        return [texts[i] for i in result], [styles[i] for i in result], [bboxes[i] for i in result]

    def get_random_color(self, wo_black=False):
        color_idx_list = copy.deepcopy(self.color_idx_list)
        if wo_black:
            color_idx_list.pop(1)
        color_name, _ = random.choice(color_idx_list)
        return color_name

    def get_random_color_list(self, num):
        color_list = random.sample(self.color_idx_list, num)
        ret_color_list = []
        for color, _ in color_list:
            ret_color_list.append(color)
        return ret_color_list

    def get_random_font(self):
        font_name, _ = random.choice(self.font_idx_list)
        return font_name

    def get_random_font_list(self, num):
        font_list = random.sample(self.font_idx_list, num)
        ret_font_list = []
        for font, _ in font_list:
            ret_font_list.append(font)
        return ret_font_list

    def clean_style(self, styles):
        for i in range(len(styles)):
            color_name = self.get_random_color()
            styles[i]['color'] = convert_name_to_hex(color_name)
        return styles

    def get_text_start_pos(self, texts):
        prompt = ""
        '''
        Text "{text}" in {color}, {type}.
        '''
        pos_list = []
        for text in texts:
            pos_list.append(len(prompt))
            text_prompt = f'Text "{text}"'

            if self.train_color_prob > 0:
                attr_list = ['0', '1']
            else:
                attr_list = ['white', 'Montserrat-Regular']

            attr_suffix = ", ".join(attr_list)
            text_prompt += " in " + attr_suffix
            text_prompt += ". "

            prompt = prompt + text_prompt
        assert len(pos_list) > 0
        pos_list.append(len(prompt))
        return pos_list

    def get_bbox_mask(self, texts, bboxes, box_idx):
        text_idx_list = self.get_text_start_pos(texts)

        # add target text box
        vision_mask_list = []
        text_mask_list = []

        for idx in box_idx:
            bbox = bboxes[idx]
            
            # box is in [x, y, w, h, angle] format
            # area of [y:y+h, x:x+w]
            vision_mask_tensor = torch.zeros(1, self.img_h, self.img_w)
            bbox = [int(v / 1645 * 224) for v in bbox]
            bbox[2] = max(bbox[2], 1)
            bbox[3] = max(bbox[3], 1)
            vision_mask_tensor[
                0, 
                bbox[1] // self.patch_size: (bbox[1] + bbox[3] + self.patch_size - 1) // self.patch_size, 
                bbox[0] // self.patch_size: (bbox[0] + bbox[2] + self.patch_size - 1) // self.patch_size
            ] = 1
            vision_mask_tensor = vision_mask_tensor.reshape((1, self.img_h * self.img_w))

            text_mask_tensor = torch.zeros(1, self.text_len)
            text_mask_tensor[
                0,
                text_idx_list[idx]: text_idx_list[idx + 1]
            ] = 1

            vision_mask_list.append(vision_mask_tensor)
            text_mask_list.append(text_mask_tensor)

        return vision_mask_list, text_mask_list

    def aug_textbox(self, text, aug_list):
        for _ in range(self.max_tries):
            strategy = random.choice(self.AUG_STRATEGIES)
            aug_text = strategy(text)
            if aug_text not in aug_list:
                return aug_text
        return None

    def __getitem__(self, index):
        try:
            meta_item = self.meta[index]
            texts = copy.deepcopy(meta_item['texts'])
            styles = copy.deepcopy(meta_item['styles'])
            bboxes = copy.deepcopy(meta_item['bbox'])

            texts, styles, bboxes = self.nms(texts, styles, bboxes)
            styles = self.clean_style(styles)

            assert len(texts) == len(styles) and len(texts) == len(bboxes)
            assert len(texts) > 0

            for i in range(len(styles)):
                if 'color' in styles[i]:
                    styles[i]['color'] = convert_name_to_hex(self.get_random_color(wo_black=True))
                font_name = self.get_random_font()
                if 'font-family' in styles[i]:
                    styles[i]['font-family'] = font_name

            _, _, font_sizes = self.get_caption_and_img(texts, styles, bboxes, auto_wrap=self.auto_wrap)

            assert len(texts) == len(font_sizes)
            text_idx = 0
            for i in range(len(font_sizes)):
                if font_sizes[i] > font_sizes[text_idx]:
                    text_idx = i

            # calculate bbox_dict
            box_idx = [i for i in range(len(texts))]
            random.shuffle(box_idx)
            box_len = min(len(texts), self.max_box_per_im)
            if text_idx not in box_idx[:box_len]:
                box_idx = [text_idx] + box_idx
            box_idx = box_idx[:box_len]

            for i in range(len(texts)):
                if isinstance(texts[i], list):
                    texts[i] = texts[i][0]

            aug_text_list = {}
            aug_color_list = {}
            aug_font_list = {}

            if self.train_color_prob > 0:
                for i in box_idx:
                    random_color_list = self.get_random_color_list(self.num_negative + 1)
                    styles[i]['color'] = convert_name_to_hex(random_color_list[0])
                    aug_color_list[i] = random_color_list[1:]

            if self.train_font_prob > 0:
                for i in box_idx:
                    random_font_list = self.get_random_font_list(self.num_negative + 1)
                    font_code = random_font_list[0]
                    styles[i]['font-family'] = font_code
                    aug_font_list[i] = random_font_list[1:]

            gt_caption, gt_img, font_sizes = self.get_caption_and_img(texts, styles, bboxes, auto_wrap=self.auto_wrap)
            if len(gt_caption) > self.text_len:
                raise ValueError
            captions = [gt_caption]
            imgs = [self.transform(gt_img)]
            vision_mask_list, text_mask_list = self.get_bbox_mask(texts, bboxes, box_idx)
            for i in box_idx:
                aug_text_list[i] = [texts[i]]
            text_attn_mask = None

            prob = random.random()
            for negative_idx in range(self.num_negative):
                aug_success = False
                for _ in range(self.max_tries):
                    if aug_success:
                        break
                    try:
                        aug_texts = copy.deepcopy(texts)
                        aug_styles = copy.deepcopy(styles)
                        if prob > self.train_color_prob + self.train_font_prob:
                            for i in box_idx:
                                aug = self.aug_textbox(aug_texts[i], aug_text_list[i])
                                if aug is None:
                                    raise ValueError
                                aug_texts[i] = aug
                        elif prob > self.train_color_prob:
                            # change font
                            for i in box_idx:
                                font_code = aug_font_list[i][negative_idx]
                                aug_styles[i]['font-family'] = font_code
                        else:
                            for i in box_idx:
                                # change color
                                aug_styles[i]['color'] = convert_name_to_hex(aug_color_list[i][negative_idx])
                        aug_caption, aug_img, _ = self.get_caption_and_img(aug_texts, aug_styles, bboxes, auto_wrap=self.auto_wrap)
                        if len(aug_caption) > self.text_len:
                            raise ValueError
                        captions.append(aug_caption)
                        imgs.append(self.transform(aug_img))
                        vision_list, text_list = self.get_bbox_mask(aug_texts, bboxes, box_idx)
                        vision_mask_list += vision_list
                        text_mask_list += text_list
                        aug_success = True
                        for i in box_idx:
                            aug_text_list[i].append(aug_texts[i])
                    except:
                        pass
                if not aug_success:
                    raise ValueError

            bbox_index = []
            reordered_vision_mask_list, reordered_text_mask_list = [], []

            for bidx in range(box_len):
                for nidx in range(self.num_negative + 1):
                    bbox_index.append(nidx)
                    reordered_vision_mask_list.append(vision_mask_list[bidx + nidx * box_len])
                    reordered_text_mask_list.append(text_mask_list[bidx + nidx * box_len])

            vision_bbox_dict = {
                'sample_per_im': self.num_negative + 1,
                'bbox_index': bbox_index,
                'bbox_list': reordered_vision_mask_list,
            }
            text_bbox_dict = {
                'bbox_index': bbox_index,
                'bbox_list': reordered_text_mask_list,
            }

            return {
                "imgs": imgs,
                "captions": captions,
                "vision_bbox_dict": vision_bbox_dict,
                "text_bbox_dict": text_bbox_dict,
                "text_attn_mask": text_attn_mask,
            }
        except:
            return self.__getitem__(np.random.randint(0, len(self.meta)))

    def __len__(self):
        return len(self.meta)
