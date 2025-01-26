import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import LlavaProcessor

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor

class LlavaDataset(Dataset):
    def __init__(self, dataset_dir:str):
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    # 读取数据集，将chat和image数据返回
    def build_dataset(self, dataset_dir:str):
        data_dir = Path(dataset_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")
        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)
    
    def __getitem__(self, index):
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return human_input, chatbot_output, image_path



# 根据processor、输入文本、输出文本、图像路径，创建编码后的数据
def build_qaimage(processor:LlavaProcessor, q_text:str, a_text:str, image_path:Path):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    raw_image = Image.open(image_path)
    inputs = processor(prompt, raw_image, return_tensors='pt')

    a_inputs_ids = processor.tokenizer(a_text, return_tensors='pt', padding="longest", truncation=True)["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_inputs_ids
    )
    return res


# 定义一个数据集加载器，将编码好的数据调整成批次，并且按分批的最长去填充padding
class TrainLlavaModelCollator:
    def __init__(self, processor:LlavaProcessor, IGNORE_INDEX:int):
        self.processor = processor
        self.ignore_index = IGNORE_INDEX
    
    # 调整模型输入与输出，输入部分是q和a的拼接，输出部分是-100和a的拼接
    def convert_one_piece(self, q_input_ids, a_input_ids):
        # q和a拼接，外加终止符
        input_ids = torch.concat(
            [q_input_ids, a_input_ids, torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)],
            axis=1
        )
        # -100和a拼接，外加终止符
        output_ids = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ignore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1)
            ], axis=1
        )
        return input_ids, output_ids
    
    def __call__(self, features):
        # 需要通过循环查看一组中最长的序列，按最长的序列填充其余短的序列的开头
        input_ids_list = []
        label_list = []
        pixel_values = []
        input_len_list = []

        for feature in features:
            # 将输入输出字符转为id，图像路径打开图像矩阵
            qaimage_output = build_qaimage(self.processor, feature[0], feature[1], feature[2])
            # 将字符id进行调整，得到模型的input与label
            temp_input_ids, temp_labels = self.convert_one_piece(qaimage_output.q_input_ids, qaimage_output.a_input_ids)

            input_ids_list.append(temp_input_ids)
            label_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)
            input_len_list.append(temp_input_ids.shape[1])   # temp_input_ids: [1, seq_len]
        
        max_input_len = max(input_len_list)

        # 遍历batch，将每个前边都补充pad至这个batch最长的长度
        final_input_ids = torch.concat([
            torch.concat([torch.full((1, max_input_len - input_len_list[index]), self.processor.tokenizer.pad_token_id), value], axis=1) 
            for index, value in enumerate(input_ids_list)
        ])
        final_label = torch.concat([
            torch.concat([torch.full((1, max_input_len - input_len_list[index]), self.processor.tokenizer.pad_token_id), value], axis=1)
            for index, value in enumerate(label_list)
        ])
        
        # 图像都是[1, 3, h, w]，直接在第一个维度进行拼接即可
        final_piexl_values = torch.concat(pixel_values, axis=0)

        # attention_mask形状和input_id一致，需要将pad_id位置上至0
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids": final_input_ids,
            "labels": final_label,
            "pixel_values": final_piexl_values,
            "attention_mask": attention_mask
        }

if __name__ == "__main__":
    data_dir = r"E:\Codes\my_llava\LLaVA-CC3M-Pretrain-595K"
    dataset = LlavaDataset(dataset_dir=data_dir)
    print(dataset[0])

    processor = LlavaProcessor.from_pretrained(r"show_model/model001")
    collator = TrainLlavaModelCollator(processor, -100)
    res = collator([dataset[0], dataset[1], dataset[2], dataset[3]])
    print(res)
