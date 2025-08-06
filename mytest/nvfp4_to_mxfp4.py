import os
import json
import yaml
import torch
import shutil
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
from vllm.model_executor.layers.quantization.utils.mxfp4_emulation_utils import to_mx
from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
HIGH_PRECISION = [torch.bfloat16, torch.float16, torch.float32]


def load_safetensors(path_model):
    model = {}
    F = safe_open(path_model, framework="pt", device='cuda')

    for k in F.keys():
        model[k] = F.get_tensor(k)

    return model

def convert_nvfp4_to_mxfp4(model_nv):
    model_mx = {}
    layers_with_scale = defaultdict(list)

    for k in model_nv.keys():
        k_list = k.split('.')

        if k_list[-1] == 'weight' and model_nv[k].dtype in HIGH_PRECISION:
            model_mx[k] = model_nv[k]
        elif k_list[-1] == 'weight_packed':
            layers_with_scale['.'.join(k_list[:-1])].append('weight_packed')
        elif k_list[-1] == 'weight_global_scale':
            layers_with_scale['.'.join(k_list[:-1])].append('weight_global_scale')
        elif k_list[-1] == 'weight_scale':
            layers_with_scale['.'.join(k_list[:-1])].append('weight_scale')

    for layer in list(layers_with_scale.keys()):
        w_nvfp4 = model_nv[f'{layer}.weight_packed']
        weight_global_scale = model_nv[f'{layer}.weight_global_scale']
        weight_block_scale = model_nv[f'{layer}.weight_scale']

        w_dq = dequantize_to_dtype(
            tensor_fp4=w_nvfp4, 
            tensor_sf=weight_block_scale,
            global_scale=weight_global_scale, 
            dtype=torch.bfloat16, 
            device=w_nvfp4.device,
            block_size = 16
            )
        
        scale, w_mxfp4 = to_mx(
                data_hp=w_dq,
                elem_dtype="fp4_e2m1",
                block_size=32,
                scaling_mode=ScaleCalculationMode.RCEIL,
            )
        
        w_mxfp4_shape = w_mxfp4.shape
        
        model_mx[f'{layer}.weight_packed'] = w_mxfp4
        model_mx[f'{layer}.weight_scale'] = scale.reshape(w_mxfp4_shape[0], int(w_mxfp4_shape[1] * 2 / 32))
    
    return model_mx

def save_safetensors_mx(output_path, safetensors, model_mx):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)  # exist_ok=True 防止目录已存在时报错
    
    save_file(model_mx, os.path.join(output_path, safetensors))

    print(f'MXFP4 model {safetensors} is save to {output_path}')

def save_index(path_model_nv, path_model_mx, safetensor_file, model_mx):
    index_path_mx = os.path.join(path_model_mx, "model.safetensors.index.json")

    if not os.path.exists(index_path_mx):
        index_path_nv = os.path.join(path_model_nv, "model.safetensors.index.json")
        with open(index_path_nv, 'r', encoding='utf-8') as f:
            index_nv = json.load(f)
        
        index_mx = {
            "metadata": index_nv["metadata"],
            "weight_map": {}
            }
        
        with open(index_path_mx, 'w', encoding='utf-8') as f:
            json.dump(index_mx, f, ensure_ascii=False, indent=4)
    
    with open(index_path_mx, 'r', encoding='utf-8') as f:
        index_mx = json.load(f)
    
    for k in model_mx.keys():
        index_mx["weight_map"][k] = safetensor_file
    
    with open(index_path_mx, 'w', encoding='utf-8') as f:
        json.dump(index_mx, f, ensure_ascii=False, indent=4)

def save_others(path_model_nv, path_model_mx):
    files_to_copy = ["tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "merges.txt", 'added_tokens.json', 'generation_config.json']

    for file_name in files_to_copy:
        src = os.path.join(path_model_nv, file_name)
        dst = os.path.join(path_model_mx, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"{file_name} is save to {path_model_mx}.")

    # config.json
    config_path_nv = os.path.join(path_model_nv, "config.json")
    config_path_mx = os.path.join(path_model_mx, "config.json")
    with open(config_path_nv, 'r', encoding='utf-8') as f:
        config = json.load(f)
        config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["group_size"] = 32
        config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["is_mx"] = True
        config["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 32
        config["quantization_config"]["config_groups"]["group_0"]["weights"]["is_mx"] = True
        config["quantization_config"]["format"] = "mxfp4-pack-quantized"
    with open(config_path_mx, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"config.json is save to {config_path_mx}.")

    # recipe.yaml
    recipe_path_nv = os.path.join(path_model_nv, "recipe.yaml")
    recipe_path_mx = os.path.join(path_model_mx, "recipe.yaml")
    with open(recipe_path_nv, 'r') as f:
        recipe = yaml.safe_load(f)
        recipe["default_stage"]["default_modifiers"]["QuantizationModifier"]["scheme"] = "MXFP4"
    with open(recipe_path_mx, 'w', encoding='utf-8') as f:
        yaml.dump(recipe, f, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)
    print(f"recipe.yaml is save to {recipe_path_mx}.")

def main():
    path_model_nv = "/data5/wzy/vLLM-Learing/mnt/ckpt/Qwen3-8B-NVFP4"
    path_model_mx = "/data5/wzy/vLLM-mxfp4/mnt/Qwen3-8B-MXFP4-debug"
    safetensor_files = list(Path(path_model_nv).glob("*.safetensors"))

    for i in range(len(safetensor_files)):
        model_nv = load_safetensors(os.path.join(path_model_nv, safetensor_files[i].name))
        model_mx = convert_nvfp4_to_mxfp4(model_nv)
        save_safetensors_mx(path_model_mx, safetensor_files[i].name, model_mx)
        save_index(path_model_nv, path_model_mx, safetensor_files[i].name, model_mx)

    save_others(path_model_nv, path_model_mx)


if __name__ == "__main__":
    main()