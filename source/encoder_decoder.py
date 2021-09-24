"""
fine-tuning the encoder-decoder BART/T5 model.
"""
import os
import torch
import pickle
import logging
import argparse

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from source.common import init_model, load_data
from source.train import evaluate, train, set_seed


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


class EncoderDecoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        print(file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        filename = f"{os.path.basename(args.model_type)}_cached_{block_size}_{filename}{'_' + args.task if args.task else ''}"
        cached_features_file = os.path.join(directory, filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Converting to token IDs")
            examples = load_data(file_path, args.task)
            logger.info(examples[:5])

            # Add prefix to the output so we can predict the first real token in the decoder
            inputs = [
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[0]))
                for ex in examples
            ]
            outputs = [
                [inputs[i][-1]]
                + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[1]))
                for i, ex in enumerate(examples)
            ]

            # Pad
            max_input_length = min(
                args.max_input_length, max([len(ex) for ex in inputs])
            )
            max_output_length = min(
                args.max_output_length, max([len(ex) for ex in outputs])
            )

            input_lengths = [min(len(ex), max_input_length) for ex in inputs]
            output_lengths = [min(len(ex), max_output_length) for ex in outputs]

            inputs = [tokenizer.encode(
                ex, add_special_tokens=False, max_length=max_input_length, pad_to_max_length=True)
                for ex in inputs]

            outputs = [tokenizer.encode(
                ex, add_special_tokens=False, max_length=max_output_length, pad_to_max_length=True)
                for ex in outputs]

            self.examples = {
                "inputs": inputs,
                "outputs": outputs,
                "input_lengths": input_lengths,
                "output_lengths": output_lengths,
            }

        logger.info(f"Saving features into cached file {cached_features_file}")
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples["input_lengths"])

    def __getitem__(self, item):
        inputs = torch.tensor(self.examples["inputs"][item])
        outputs = torch.tensor(self.examples["outputs"][item])

        max_length = inputs.shape[0]
        input_lengths = self.examples["input_lengths"][item]
        input_mask = torch.tensor([1] * input_lengths + [0] * (max_length - input_lengths))

        max_length = outputs.shape[0]
        output_lengths = self.examples["output_lengths"][item]
        output_mask = torch.tensor([1] * output_lengths + [0] * (max_length - output_lengths))
        
        return {
            "inputs": inputs,
            "input_mask": input_mask,
            "outputs": outputs,
            "output_mask": output_mask,
        }


def get_loss(args, batch, model):
    """
    Compute this batch loss
    """
    input_ids = batch["inputs"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)
    target_ids = batch["outputs"].to(args.device)
    output_mask = batch["output_mask"].to(args.device)
    decoder_input_ids = target_ids[:, :-1].contiguous()

    # We don't send labels to model.forward because we want to compute per token loss
    lm_logits = model(
        input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, use_cache=False
    )[0]  # use_cache=false is added for HF > 3.0
    batch_size, max_length, vocab_size = lm_logits.shape

    # Compute loss for each instance and each token
    loss_fct = CrossEntropyLoss(reduction="none")
    lm_labels = target_ids[:, 1:].clone().contiguous()
    lm_labels[target_ids[:, 1:] == args.pad_token_id] = -100
    loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(
        batch_size, max_length
    )

    # Only consider non padded tokens
    loss_mask = output_mask[..., :-1].contiguous()
    loss = torch.mul(loss_mask, loss)  # [batch_size, max_length]
    return loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        required=True,
        help="Out directory for checkpoints.",
    )

    # Other parameters
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--eval_batch_size", default=64, type=int, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        required=True,
        help="The input CSV validation file."
    )
    parser.add_argument(
        "--eval_during_train",
        action="store_true",
        help="Evaluate at each train logging step.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-6,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=-1,
        help="Log every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--max_input_length",
        default=140,
        type=int,
        help="Maximum input event length in words.",
    )
    parser.add_argument(
        "--max_output_length",
        default=120,
        type=int,
        help="Maximum output event length in words.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: total number of training steps to perform.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bart-large",
        type=str,
        help="LM checkpoint for initialization.",
    )
    parser.add_argument(
        "--model_type",
        default="",
        type=str,
        help="which family of LM, e.g. gpt, gpt-xl, ....",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=2.0,
        type=float,
        help="Number of training epochs to perform.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached data."
    )
    parser.add_argument(
        "--overwrite_out_dir",
        action="store_true",
        help="Overwrite the output directory.",
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from the last checkpoint.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=-1,
        help="Save checkpoint every X updates steps (default after each epoch).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization."
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        help="The input CSV train file."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="what is the task?"
    )
    args = parser.parse_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(args.out_dir)
        and len(os.listdir(args.out_dir)) > 1
        and args.do_train
        and not args.overwrite_out_dir
        and not args.continue_training
    ):
        raise ValueError(
            f"Output directory {args.out_dir} already exists and is not empty. "
            f"Use --overwrite_out_dir or --continue_training."
        )

    # Setup device
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    # Set seed
    set_seed(args)

    # Load the models
    if args.continue_training:
        args.model_name_or_path = args.out_dir
    # Delete the current results file
    else:
        eval_results_file = os.path.join(args.out_dir, "eval_results.txt")
        if os.path.exists(eval_results_file):
            os.remove(eval_results_file)

    args.device = "cpu"
    tokenizer, model = init_model(
        args.model_name_or_path, device=args.device, do_lower_case=args.do_lower_case, args = args
    )

    args.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Pad token ID: {args.pad_token_id}")
    args.block_size = tokenizer.max_len_single_sentence
    logger.info(f"Training/evaluation parameters {args}")

    eval_dataset = None
    if args.do_eval or args.eval_during_train:
        eval_dataset = EncoderDecoderTextDataset(
            tokenizer, args, file_path=args.eval_data_file, block_size=args.block_size)

    # Add special tokens (if loading a model before fine-tuning)
    if args.do_train and not args.continue_training:
        special_tokens = ["[shuffled]", "[orig]", "<eos>"]
        extra_specials = [f"<S{i}>" for i in range(args.max_output_length)]
        special_tokens += extra_specials
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "<eos>"
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    args.pad_token_id = tokenizer.pad_token_id

    # resize_token_embeddings for Bart doesn't work if the model is already on the device
    args.device = device
    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = EncoderDecoderTextDataset(
            tokenizer,
            args,
            file_path=args.train_file,
            block_size=args.block_size,
        )
        global_step, tr_loss = train(
            args,
            train_dataset,
            model,
            tokenizer,
            loss_fnc=get_loss,
            eval_dataset=eval_dataset,
        )
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

        # Create output directory if needed
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        logger.info(f"Saving model checkpoint to {args.out_dir}")

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.out_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        tokenizer, model = init_model(
            args.out_dir, device=args.device, do_lower_case=args.do_lower_case, args=args
        )
        args.block_size = tokenizer.max_len_single_sentence
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint = args.out_dir
        logger.info(f"Evaluate the following checkpoint: {checkpoint}")
        prefix = (
            checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        )
        _, model = init_model(
            checkpoint, device=args.device, do_lower_case=args.do_lower_case, args=args
        )

        model.to(args.device)
        result = evaluate(eval_dataset, args, model, prefix=prefix, loss_fnc=get_loss)
        results.update(result)

    return results


if __name__ == "__main__":
    main()
