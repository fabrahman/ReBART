import os
import json
import pandas as pd
import torch.nn as nn
from transformers import AutoModelWithLMHead, AutoTokenizer


def init_model(model_name: str, device, do_lower_case: bool = False, args=None):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)

    # uncomment for using data parallel
#    special_tokens = ["[shuffled]", "[orig]", "<eos>"]
#    extra_specials = [f"<S{i}>" for i in range(args.max_output_length)]
#    special_tokens += extra_specials
#    tokenizer.pad_token = "<pad>"
#    tokenizer.eos_token = "<eos>"
#    tokenizer.add_tokens(special_tokens)
#
#    model.resize_token_embeddings(len(tokenizer))
#    model = nn.DataParallel(model, device_ids = [1, 2])
    model.to(device)
    model.eval()
    return tokenizer, model


### Reordering task
def load_data(in_file, task="in_shuf"):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """
    all_lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))
    if task == "index_with_sep":
        examples = [
            (
                f"[shuffled] {' '.join([' '.join((f'<S{i}>', sent)) for i, sent in zip(list(range(len(line['orig_sents']))), line['shuf_sents'])])} [orig]",
                f"{' '.join(line['orig_sents'])} <eos>",
            )
        for line in all_lines
    ]
    else:
        examples = [
            (
                f"[shuffled] {line['shuf_sents'].rstrip(' <eos>') if type(line['shuf_sents']) == str else ' '.join(line['shuf_sents'])} [orig]",
                f"{line['orig_sents'].rstrip(' <eos>') if type(line['orig_sents']) == str else ' '.join(line['orig_sents'])} <eos>",
            )
            for line in all_lines
        ]
    return examples
