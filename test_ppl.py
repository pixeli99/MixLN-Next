import os
import math
import time
import argparse

import torch
import torch.nn.functional as F

import datasets
from tqdm import tqdm
# from safetensors.torch import load_file

from transformers import AutoConfig, AutoTokenizer
# 如果你在训练时使用的是自定义的 LlamaForCausalLM（比如你在代码中提到的 peft_pretraining.modeling_llama），
# 那么这里需要和训练时的模型类对应。假设是以下：
from peft_pretraining.modeling_llama import LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型文件所在的文件夹，内部应该包含 model.safetensors、config.json 等。")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理所使用的设备，如 'cuda' 或者 'cpu'。")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="推理的精度，如 'bfloat16' 或者 'float32'。若 GPU 不支持 bf16，请改为 float32。")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="验证时的 batch size。")
    parser.add_argument("--max_length", type=int, default=256, 
                        help="tokenizer 的最大长度，与训练时一致。")
    parser.add_argument("--max_eval_samples", type=int, default=100, 
                        help="从验证集前面截取多少条用于评估，过大可能占用显存或运行慢。")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                        help="验证集路径，可直接指定 HF 路径或者本地路径。例如 '/xxx/c4/en'。")
    return parser.parse_args()

@torch.no_grad()
def compute_ppl(model, tokenizer, args):
    """
    在指定的验证数据集上计算平均 PPL
    """
    # 这里默认你使用的是 C4 的 validation split，如需换别的请自行修改
    # 如果你是从本地/disk加载数据，或者 streaming=False，请相应修改。
    if args.eval_dataset_path is None:
        # 例如 c4 路径可以是 "c4/en"
        raise ValueError("请指定 --eval_dataset_path，例如 '/lpai/.../c4/en' 或者 streaming=False 的路径")
    
    # 载入数据（默认 streaming=True）
    ds = datasets.load_dataset(
        args.eval_dataset_path, 
        split="validation", 
        streaming=True
    )
    
    # 截取 max_eval_samples 条用来评估
    ds = ds.take(args.max_eval_samples)

    # 设置模型为 eval 模式
    model.eval()

    # 记录总 loss 与样本条数
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(ds, total=args.max_eval_samples, desc="Evaluating")

    # 累积 batch，用于 batch 推理
    text_batch = []
    for idx, example in enumerate(pbar):
        text_batch.append(example["text"])
        # 当积累的样本数达到 batch_size 或者到了末尾，就进行一次 forward
        if len(text_batch) == args.batch_size:
            _loss, _tokens = forward_batch(model, tokenizer, text_batch, args)
            total_loss += _loss
            total_tokens += _tokens
            text_batch = []

    # 处理最后一个不满 batch 的情况
    if len(text_batch) > 0:
        _loss, _tokens = forward_batch(model, tokenizer, text_batch, args)
        total_loss += _loss
        total_tokens += _tokens

    # 平均 loss（注意这里除以总 token 数，以便和通常语言模型 ppl 定义一致）
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return ppl

@torch.no_grad()
def forward_batch(model, tokenizer, text_list, args):
    """
    对一批文本进行推理，返回累计的 total_loss 和该 batch 里面的 total_tokens。
    """

    # 把文本转成 input_ids
    batch = tokenizer(
        text_list,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 模型要求的 inputs
    input_ids = batch["input_ids"].to(args.device)
    attention_mask = batch["attention_mask"].to(args.device)

    # 计算 tokens 的有效数量 (不包含 pad)
    valid_tokens = attention_mask.sum().item()

    # labels
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    total_loss = loss.item() * valid_tokens  # 累计 loss

    return total_loss, valid_tokens

def main():
    args = parse_args()
    device = torch.device(args.device)

    # ============== 1. 加载模型配置 ==============
    # 从 model_dir 中读取 config
    config_path = os.path.join(args.model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到 config.json，路径：{config_path}")
    config = AutoConfig.from_pretrained(config_path)
    print("Loaded model config.")

    # ============== 2. 初始化模型 ==============
    # 注意，这里我们使用自定义的 LlamaForCausalLM，如果你保存时是另外一个模型结构，要保持一致。
    model = LlamaForCausalLM(config)
    
    # 从 safetensors 加载权重
    weights_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        # 若你保存的是 pytorch_model.bin，可以改成下面这样：
        # weights_path = os.path.join(args.model_dir, "pytorch_model.bin")
        raise FileNotFoundError(f"找不到模型权重，路径：{weights_path}")
    
    print("Loading weights from:", weights_path)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict, strict=True)
    print("Model weights loaded successfully.")
    
    # 移动到指定的 device，并设置 dtype
    if args.dtype.lower() in ["bf16", "bfloat16"] and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    elif args.dtype.lower() in ["fp16", "float16"]:
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)

    # ============== 3. 加载分词器 ==============
    # 这里用一个示例，如训练时是用 "t5-base" (只要跟训练一致就行)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", use_fast=False)
    # 如果是 Llama tokenizer，需要对应:
    # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=False)
    
    # 有些 tokenizer 可能没有 pad_token，需要手动加一个 pad_token
    # 如果你的 tokenizer 在训练时就已经有 pad_token，就不需要这一步
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    # ============== 4. 计算 PPL ==============
    start_time = time.time()
    ppl = compute_ppl(model, tokenizer, args)
    end_time = time.time()

    print(f"Validation PPL = {ppl:.4f}")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    rec_list = []
    for idx in range(24):
        print(idx, model.model.layers[idx].rec_iter)
        rec_list.append(model.model.layers[idx].rec_iter)
    # draw rec_iter
    import matplotlib.pyplot as plt
    plt.plot(rec_list)
    plt.savefig('rec_iter.png')

if __name__ == "__main__":
    main()