import json

# 从.json文件中读取数据
with open(r'E:\comprehensive_library\e_commerce_lmm\data\openi-zh-prompt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 数据格式转换
new_data = []
for d in data:
    new_conversations = dict()
    new_conversations["conversations"] = [
        {
            "from": "user",
            "value": f'Picture 1: <img>{d["img"]}</img>\n{d["prompt"]}'
        },
        {
            "from": "assistant",
            "value": d["label"]
        }
    ]
    new_data.append(new_conversations)

# 将转换后的数据写入新的.json文件中
with open('../data/openai-zh-swift-qwenvl-prompt.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)