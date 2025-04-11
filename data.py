import json

# 加载原始数据
with open("financial_statements_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 构造 Qwen 格式
formatted = []
for i in data:
    question = f"请分析如下财务信息：{i['input']}"
    cot = "思考过程：先识别财务类型，再从资产、负债、营收等方面逐条分析，最终形成结论。"
    response = i['output']

    formatted.append({
        "Question": question,
        "Complex_CoT": cot,
        "Response": response
    })

# 保存为 jsonl
with open("qwen_financial_data.jsonl", "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ 转换完成，生成 qwen_financial_data.jsonl")

