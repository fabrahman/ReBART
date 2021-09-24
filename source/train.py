"""
Based on https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py:
fine-tuning language models on a text file using a causal language modeling (CLM) loss.
"""
import os
import re
import glob
import torch
import random
import shutil
import pickle
import logging
import argparse
import numpy as np

from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from source.common import init_model, load_data


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def set_seed(args):
    """
    Set the random seed for reproducibility
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_loss(args, batch, model):
    """
    Compute this batch loss
    """
    token_ids = batch["examples"].to(args.device)
    input_mask = batch["input_mask"].to(args.device)

    # We don't send labels to model.forward because we want to compute per token loss
    lm_logits = model(token_ids, attention_mask=input_mask)[0]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    batch_size, max_length, vocab_size = shift_logits.shape

    # Compute loss for each instance and each token
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = token_ids[..., 1:].contiguous().view(-1)
    loss = loss_fct(shift_logits, shift_labels).view(batch_size, max_length)

    # Only consider non padded tokens
    loss_mask = input_mask[..., :-1].contiguous()
    loss = torch.mul(loss_mask, loss)  # [batch_size, max_length]

    return loss

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """
    Keep a maximum of args.save_total_limit checkpoints.
    """
    if not args.save_total_limit:
        return

    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(args.out_dir, "{}-*".format(checkpoint_prefix))
    )
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer, loss_fnc=get_loss, eval_dataset=None):
    """
    Train the model.
    """
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    # Set the number of steps based on the num_epochs * len(train) or args.max_steps if specified.
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist and load from there
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    # Train
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(
                f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch"
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            # Take the loss only for the part after the input (as in seq2seq architecture)
            loss = loss_fnc(args, batch, model)
            loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.eval_during_train:
                        results = evaluate(eval_dataset, args, model, loss_fnc=loss_fnc)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"

                    # Save model checkpoint
                    out_dir = os.path.join(
                        args.out_dir, "{}-{}".format(checkpoint_prefix, global_step)
                    )

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    torch.save(args, os.path.join(out_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", out_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(
                        optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(out_dir, "scheduler.pt")
                    )
                    logger.info("Saving optimizer and scheduler states to %s", out_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(eval_dataset, args, model, prefix="", loss_fnc=get_loss):
    """
    Evaluation
    """
    eval_out_dir = args.out_dir

    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    micro_loss = macro_loss = 0.0
    num_tokens = num_batches = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            batch_loss = loss_fnc(args, batch, model)
            macro_loss += batch_loss.mean().item()
            micro_loss += batch_loss.sum().item()
            num_tokens += batch_loss.view(-1).shape[0]
        num_batches += 1

    macro_perplexity = torch.exp(torch.tensor(macro_loss / num_batches))
    micro_perplexity = torch.exp(torch.tensor(micro_loss / num_tokens))

    result = {
        "macro_perplexity": macro_perplexity,
        "micro_perplexity": micro_perplexity,
    }

    output_eval_file = os.path.join(eval_out_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** Eval results {prefix} *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {result[key]}")
            writer.write(f"{key} = {result[key]}\n")

    return result


if __name__ == "__main__":
    main()
