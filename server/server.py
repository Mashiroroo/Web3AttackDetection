from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# 确定设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 本地模型和分词器的路径
model_path = "./Qwen2-1.5B-Instruct"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 将模型移到 cuda:0
model.to(device)


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # 应用聊天模板并生成输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成 attention_mask
    attention_mask = model_inputs['input_ids'].ne(tokenizer.pad_token_id).long().to(device)

    # 生成文本
    with torch.no_grad():  # 禁用梯度计算以节省内存
        generated_ids = model.generate(
            input_ids=model_inputs['input_ids'],
            attention_mask=attention_mask,
            max_new_tokens=512
        )

    # 处理生成的文本
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
    ]

    # 解码并返回生成的文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(port=50000, debug=True)
