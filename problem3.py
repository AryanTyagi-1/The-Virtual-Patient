# virtual_patient_finetune.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training



MODEL_NAME = "decapoda-research/llama-7b-hf"  

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
)


model = prepare_model_for_int8_training(model)



lora_config = LoraConfig(
    r=16,               
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")



def load_scripts(file_path):
    """Load synthetic doctor-patient conversation data."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def format_example(persona, doctor_utterance, patient_response):
    """Format example for causal language model fine-tuning; includes persona for conditioning."""
    prompt = f"[Patient Persona: {persona}]\nDoctor: {doctor_utterance}\nPatient:"
    input_text = f"{prompt} {patient_response}"
    return input_text

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        texts = []
        for entry in data:
            persona = entry["persona"]
            for conv in entry["dialogues"]:
                texts.append(format_example(persona, conv["doctor"], conv["patient"]))
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


DATA_PATH = "synthetic_doctor_patient.json"

print("Loading and processing dataset...")
dataset = load_scripts(DATA_PATH)
train_dataset = PatientDataset(dataset)




training_args = TrainingArguments(
    output_dir="./patient_model",
    per_device_train_batch_size=1,   
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,                       
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting fine-tuning...")
trainer.train()

print("Saving fine-tuned model...")
model.save_pretrained("./patient_model_lora")
tokenizer.save_pretrained("./patient_model_lora")



from transformers import pipeline

print("Loading fine-tuned model for inference...")

model = AutoModelForCausalLM.from_pretrained("./patient_model_lora", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("./patient_model_lora")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_patient_response(persona, doctor_input):
    prompt = f"[Patient Persona: {persona}]\nDoctor: {doctor_input}\nPatient:"
    generated = generator(prompt, max_length=150, do_sample=True, temperature=0.8, num_return_sequences=1)
    reply = generated[0]['generated_text'].split("Patient:")[-1].strip()
    return reply


print("Example generations:\n")

for persona in ["calm", "anxious"]:
    doctor_question = "How are you feeling today?"
    patient_reply = generate_patient_response(persona, doctor_question)
    print(f"Persona: {persona.capitalize()}")
    print(f"Doctor: {doctor_question}")
    print(f"Patient: {patient_reply}\n")
