import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import os
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
import transformers
from trl import SFTTrainer


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "google/gemma-7b-it"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

classified_shrishti_df = pd.read_csv('/home/jupyter/finetune_LLM/Data/shrishti/train/processed_chunked_files_llm_finetuining_shrishti.csv')

# Define the instruction string
instruction = '''Classify the input into below categories:
- recipere
- investigations
- plan
- complaints
- history_of_previous_illness
- examination
- diagnoses
Output the classified data into JSON format'''

# Add the 'instruction' column with the repeated string
classified_shrishti_df['instruction'] = instruction

# Rename columns
new_column_names = {
    'notes': 'input',
    'classified': 'output',
}
classified_shrishti_df = classified_shrishti_df.rename(columns=new_column_names)

dataset = Dataset.from_pandas(classified_shrishti_df)

def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'You are a Medical Clinical notes document entity extraction specialist.' \
               'You are also provided with the Doctors Note.\n\n'
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in dataset]
dataset = dataset.add_column("prompt", text_column)

dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

dataset = dataset.train_test_split(test_size=0.2)
train_data = dataset["train"]
test_data = dataset["test"]

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        #warmup_steps=0.03,
        max_steps=1000,
        learning_rate=2e-4,
        logging_steps=100,
        output_dir="./finetune_LLM/finetuned_LLM/gemma_shrishti_outputs/",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

new_model = "./gemma-Code-Instruct-Finetune-test"

trainer.model.save_pretrained(new_model)

model_id = "google/gemma-7b-it"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./finetune_LLM/finetuned_LLM/gemma_shrishti_outputs/merged_model",safe_serialization=True)
tokenizer.save_pretrained("./finetune_LLM/finetuned_LLM/gemma_shrishti_outputs/merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

