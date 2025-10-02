from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    device_map="cuda",
    dtype="auto",
    trust_remote_code=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct")

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer, 
    return_full_text = False,
    max_new_tokens = 500,
    do_sample = False,
)

# The prompt (user input / query)
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])

