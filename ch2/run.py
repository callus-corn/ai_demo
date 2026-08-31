from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils.logging import disable_progress_bar


disable_progress_bar()

# モデルとトークナイザーの読み込み
tokenizer = GPT2Tokenizer.from_pretrained("./", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("./", local_files_only=True)

# 入力
prompt = "Hello"

# トークン生成
inputs = tokenizer(prompt, return_tensors="pt")

# 生成
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)

# デコード
generate_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_text)
