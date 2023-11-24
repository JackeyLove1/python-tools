from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import pipelines
def main():
    model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    load_in_4bit = False
    max_new_tokens = 512
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    #TODO: support quantization_config used by BitsAndBytesConfig
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).eval()
    #TODO: in torch2.0 use torch.compile
    #TODO: support quantization_config used by BitsAndBytesConfig
    while True:
        text = input("User: ")
        text = text.strip()
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        bos_token_id = torch.tensor([[tokenizer.bos_token_id]],dtype=torch.long).to(device)
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("AI：{}".format(response))