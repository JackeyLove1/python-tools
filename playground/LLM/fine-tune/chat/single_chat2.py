from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
from transformers import pipeline


def main():
    model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    pipe = pipeline("text-generation", model=model_name_or_path, torch_dtype=torch.bfloat16,
                    device_map="auto")
    while True:
        text = input("User: ")
        text = text.strip()
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": text},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print("AI:", outputs[0]["generated_text"])
