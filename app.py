import os
os.system("mim install mmcv-full")
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download internlm2 to the base_path directory using git tool
base_path = './personal_assistant'
os.system(f'git clone https://code.openxlab.org.cn/sunxiaobei/personal_assistant.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="小煤球",
                description="""
个人安全智能小助手，提供安全相关的问题解答和建议。如果有任何问题，请随时问我！  
                 """,
                 ).queue(1).launch()

