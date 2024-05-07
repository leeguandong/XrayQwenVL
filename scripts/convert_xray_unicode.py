import json
from tqdm import tqdm

if False:
    # 从文件中读取数据
    with open(r'E:\comprehensive_library\e_commerce_lmm\data\openi-zh.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 将json数据写入新的json文件
    with open('../data/open-zh-utf8.json', 'w', encoding='utf-8') as f:
        # 确保ascii设置为False以保存unicode字符
        json.dump(json_data, f, ensure_ascii=False, indent=4)

if True:
    with open(r'E:\comprehensive_library\e_commerce_lmm\data\open-zh-utf8.json', encoding="utf-8") as f:
        data = json.load(f)

    data_info = []
    for i in tqdm(range(len(data['annotations']))):
        img = data['annotations'][i]['image_id']
        prompt = '通过这张胸部X光影像可以诊断出什么？'
        label = data['annotations'][i]['caption']
        json_data = {
            'img': "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/XrayGLM/images2/" + str(
                img) + '.png',
            'prompt': prompt,
            'label': str(label)
        }
        data_info.append(json_data)

    with open('../data/openi-zh-prompt.json', 'w+', encoding="utf-8") as f1:
        json.dump(data_info, f1, ensure_ascii=False, indent=4)
