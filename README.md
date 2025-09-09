# The-Virtual-Patient
The Government of India wants to help medical students in rural areas who don’t always have access to real patients. They are investing in a VR training system. A student puts on a headset, and suddenly they are sitting across the table from a patient. The patient might be calm, anxious, rude, or overly patient - just like in real life.
# Virtual Patient Dialogue Generator
Imagine training a friendly, realistic virtual patient who can talk with medical students just like a real person — calm, anxious, rude, or patient — right from your own laptop or a small device. This project makes that possible using lightweight, quantized LLaMA models with Ollama.
# What we  Need
* A computer with at least 8GB RAM (more is better for bigger models)
* Ollama installed on your machine: Ollama lets you run large language models locally without fuss.
* Python 3.8+ for running fine-tuning scripts (latest is best).
* Your dataset of synthetic doctor-patient conversations.
* OpenAI API key (optional, if you want to compare or combine with cloud models)
# Setup or Run instructions
1. Install Ollama
Head over to ollama.com and download the installer for your OS. They make it super simple — just install and you’re done.
After installing, open a terminal and check it’s working:
bash
ollama --version
You should see the version info pop up.
2. Pull Your Base Model
For lightweight use, grab a model like Gemma or LLaMA 7B:
bash
ollama pull gemma:latest
 or
ollama pull llama7b:latest.
This downloads and prepares the model locally so you can fine-tune or run it.
3. Prepare Your Training Data
Format your conversation scripts into JSON or text file with clear labeling of patient personas and turns, for example:
json
{
  "persona": "anxious",
  "dialogues": [
    {"doctor": "How are you feeling today?", "patient": "I'm a bit nervous about the tests."},
    ...
  ]
}
Keep it consistent so the model learns different personalities well.
4. Fine-tune Using LoRA (Recommended for Efficient Training)
  Install some Python dependencies:
 bash
 pip install peft transformers accelerate.
5. Run Your Virtual Patient Locally
Start the model in Ollama:
bash
ollama run your-finetuned-model
You can chat with the patient in different personas by providing persona prompts, e.g.
"You are an anxious patient. Respond to the doctor's questions accordingly."
6.  Integrate with VR or Voice Systems
Use Ollama's API or CLI to feed text prompts and get model responses back in real-time. This way, your VR system's patient avatar can speak and react naturally.

