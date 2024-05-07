import json

# 打开并加载.json文件
with open(r"E:\comprehensive_library\e_commerce_lmm\data\openi-zh-prompt.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 数据格式转换
new_data = []
for i, d in enumerate(data):
    new_conversations = dict()
    new_conversations["id"] = f"identity_{i}"
    new_conversations["conversations"] = [
        {
            "from": "user",
            "value": "Picture 1: <img>" + d["img"] + "</img>\n" + d["prompt"]
        },
        {
            "from": "assistant",
            "value": d["label"]
        }
    ]
    new_data.append(new_conversations)

# 将新的数据写入新的.json文件中
with open("../data/openai-zh-qwenvl-prompt.json", 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
