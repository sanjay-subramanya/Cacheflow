from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "models/base"

# 1. Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

# 2. Define your conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "what is the future of AI?."}
]

# 3. Prepare the prompt using the model's specific chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 4. Generate the response
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

# 5. Decode and print
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response.split("assistant")[-1].strip())