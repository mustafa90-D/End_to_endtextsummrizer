from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_from_disk
import os
from src.textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_ckpt = self.config.model_ckpt.replace("\\", "/")

        if os.path.isdir(model_ckpt):  # Local model directory
            from pathlib import Path

            tokenizer_path = Path(self.config.tokenizer_path).resolve()
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
        else:  # Hugging Face model ID
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=int(1e6),
            gradient_accumulation_steps=16
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model and tokenizer
        model_save_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_save_path = os.path.join(self.config.root_dir, "tokenizer")

        model_pegasus.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)


    # def train(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     from pathlib import Path

    #     # Force it to use as a local path
    #     model_ckpt_path = Path(self.config.model_ckpt).resolve()
    #     tokenizer = AutoTokenizer.from_pretrained(str(model_ckpt_path), local_files_only=True)

    #     # tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
    #     model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
    #     seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

    #     #loading the data
    #     dataset_samsum_pt = load_from_disk(self.config.data_path)

    #     trainer_args = TrainingArguments(
    #         output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
    #         per_device_train_batch_size=1, per_device_eval_batch_size=1,
    #         weight_decay=0.01, logging_steps=10,
    #         evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    #         gradient_accumulation_steps=16
    #     ) 
    #     trainer = Trainer(model=model_pegasus, args=trainer_args,
    #               tokenizer=tokenizer, data_collator=seq2seq_data_collator,
    #               train_dataset=dataset_samsum_pt["test"],
    #               eval_dataset=dataset_samsum_pt["validation"])
        
    #     trainer.train()

    #     ## Save model
    #     model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
    #     ## Save tokenizer
    #     tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


