import base64
import os
import json
from io import BytesIO
from multiprocessing import Pool, Manager
from PIL import Image, ImageFile
from tqdm import tqdm

img_folder = "/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/ECommerce-IC/png_images"

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

images = {}
with open("/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/ECommerce-IC/IC_train.tsv", "r",
          encoding="utf-8") as tsv_file:
    for line in tsv_file:
        img_id, img_content = line.strip().split("\t")
        images[img_id] = img_content


def process_line(line):
    try:
        entry = json.loads(line)
        image_id = entry["image_id"]
        text = " ".join(entry["text"])
        img_content = images[image_id]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(BytesIO(base64.urlsafe_b64decode(img_content)))
        img_path = os.path.join(img_folder, f"{image_id}.png")
        img.save(img_path)
        convo = {
            "conversations": [
                {"from": "user", "value": f"Picture 1:<img>{img_path}</img>\n这张图描述了什么？"},
                {"from": "assistant", "value": f"{text}"}
            ]
        }
        result_queue.put(convo)
    except OSError:
        print(f"Skipping the data for image ID: {image_id} due to an OSError.")


if __name__ == "__main__":
    with Manager() as manager:
        result_queue = manager.Queue()
        with open("/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/ECommerce-IC/IC_train.jsonl", "r",
                  encoding="utf-8") as json_file:
            lines = json_file.readlines()
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
                pass
        conversations = [result_queue.get() for _ in range(result_queue.qsize())]
        # print(conversations)

    with open('/home/image_team/image_team_docker_home/lgd/e_commerce_lmm/data/ECommerce-IC/IC_train.json', 'w',
              encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
