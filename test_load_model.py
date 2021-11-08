import os
import requests
import jax

from koclip import load_koclip_custom

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model, processor = load_koclip_custom("koclip-base", cache_dir='koclip_model')