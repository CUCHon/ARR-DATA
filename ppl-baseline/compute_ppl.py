# compute_ppl_batched.py

import os
import json
import math
import torch
import argparse
from glob import glob
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def compute_ppl_batched(
    texts, tokenizer, model, device,
    max_length=1024, batch_size=8
):

    all_ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=tokenizer.pad_token_id
    )

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids      = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=None
            )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask   = attention_mask[..., 1:].contiguous().float()

        loss_flat = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(input_ids.size(0), -1)

        sum_loss = (loss_flat * shift_mask).sum(dim=1)
        token_cnt= shift_mask.sum(dim=1)
        avg_nll  = sum_loss / token_cnt
        ppls     = torch.exp(avg_nll).cpu().tolist()

        all_ppls.extend(ppls)

        del enc, input_ids, attention_mask, outputs, logits
        torch.cuda.empty_cache()

    return all_ppls

def process_file(path, tokenizer, model, device, template, batch_size):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [
        template.format(instruction=s.get("instruction",""),
                        input=s.get("input","")) + s.get("output","")
        for s in data
    ]

    ppl_list = compute_ppl_batched(
        texts, tokenizer, model, device,
        max_length=1024, batch_size=batch_size
    )
    for s, ppl in zip(data, ppl_list):
        s["ppl"] = ppl

    base, ext = os.path.splitext(path)
    out_path  = f"{base}_with_ppl{ext}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ {os.path.basename(out_path)} ")

def main():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-i","--inputs", required=True, nargs="+",
        help=""
    )
    parser.add_argument(
        "-m","--model", default="gpt2",
        help=""
    )
    parser.add_argument(
        "-b","--batch_size", type=int, default=8,
        help=""
    )
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n{input}\n\n### Response:"
    )

    files = sorted({p for pattern in args.inputs for p in glob(pattern)})
    for path in files:
        process_file(
            path, tokenizer, model, device,
            template, args.batch_size
        )

if __name__ == "__main__":
    main()
