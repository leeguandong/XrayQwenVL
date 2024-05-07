import json
import torch
import os
from transformers import AutoTokenizer, AutoModel


def merge_lora(pt1, pt2):
    from finetune_visualglm import FineTuneVisualGLMModel
    import argparse

    model, args = FineTuneVisualGLMModel.from_pretrained(pt1,
                                                         args=argparse.Namespace(
                                                             fp16=True,
                                                             skip_init=True,
                                                             use_gpu_initialization=True,
                                                         ))
    model.get_mixin('lora').merge_lora()
    args.layer_range = []
    args.save = pt2
    with open(os.path.join(pt1, 'latest')) as f:
        latest = f.readlines()[0].strip()
    merge_pt_file = os.path.join(pt2, latest, 'mp_rank_00_model_states.pt')
    args.mode = 'inference'
    from sat.training.model_io import save_checkpoint
    save_checkpoint(latest, model, None, None, args)
    return merge_pt_file


def pt_hf(outputhf, pt, index_file):
    index_dict = json.load(open(index_file))
    pt = torch.load(pt, map_location="cpu")["module"]
    if not os.path.exists(outputhf):
        os.mkdir(outputhf)
    for index in range(5):
        filename = f"pytorch_model-{index + 1:05}-of-00005.bin"
        state_dict = {}
        params = [i for i, j in index_dict['weight_map'].items() if j == filename]
        for hf_param in params:
            if hf_param.endswith('attention.rotary_emb.inv_freq'):
                state_dict[hf_param] = torch.tensor(
                    [1.0000e+00, 7.5000e-01, 5.6250e-01, 4.2163e-01, 3.1616e-01, 2.3718e-01,
                     1.7786e-01, 1.3330e-01, 9.9976e-02, 7.5012e-02, 5.6244e-02, 4.2175e-02,
                     3.1616e-02, 2.3712e-02, 1.7776e-02, 1.3336e-02, 1.0002e-02, 7.4997e-03,
                     5.6229e-03, 4.2152e-03, 3.1624e-03, 2.3708e-03, 1.7786e-03, 1.3332e-03,
                     1.0004e-03, 7.5006e-04, 5.6219e-04, 4.2176e-04, 3.1614e-04, 2.3711e-04,
                     1.7786e-04, 1.3340e-04], device='cuda:0', dtype=torch.float16)
            elif hf_param.startswith('transformer.layers') and 'attention.query_key_value.bias' in hf_param:
                state_dict[hf_param] = pt[hf_param].view(3, 32, 128).permute(1, 0, 2).reshape(12288)
            elif hf_param.startswith('transformer.layers') and 'attention.query_key_value.weight' in hf_param:
                state_dict[hf_param] = pt[hf_param].view(3, 32, 128, 4096).permute(1, 0, 2, 3).reshape(12288, 4096)
            else:
                pt_param = hf_param.replace('image_encoder', 'mixins.eva.model').replace('lm_head.weight',
                                                                                         'mixins.chatglm-final.lm_head.weight')
                state_dict[hf_param] = pt[pt_param]
        torch.save(state_dict, os.path.join(outputhf, filename))
        print(filename, 'success save', len(state_dict), '\n')


def test_model(hf):
    tokenizer = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf, trust_remote_code=True).half().cuda()
    image_path = "test.png"
    response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
    print(response)
    response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=[])
    print(response)


if __name__ == '__main__':
    pt_ft = 'finetune-visualglm-6b-09-10-23-45'
    pt_ft_merge = 'finetune-visualglm-6b-09-10-23-45_merge'
    hf_new = 'finetune-visualglm-6b-09-10-23-45_hf'
    latest = '1'
    merge_pt_file = f'finetune-visualglm-6b-09-10-23-45_merge/{latest}/mp_rank_00_model_states.pt'
    index_file = './visualglm-6b/pytorch_model.bin.index.json'
    merge_lora(pt_ft, pt_ft_merge)
    pt_hf(hf_new, merge_pt_file, index_file)

    # 测试生成的hf是否可用,生成的bin文件替换掉https://huggingface.co/THUDM/visualglm-6b/tree/main里的bin文件，然后就可以load了
    test_model(hf_new)