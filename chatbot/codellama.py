import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# BitsAndBytesConfig 설정
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype="float16",
)

# codellama 모델 불러오기
@st.cache_resource
def load_model(base_model_path, peft_model_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_path)
    return model

# codellama 토크나이저 불러오기
@st.cache_data
def load_tokenizer(base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return tokenizer

# 기존의 프롬프트
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "

# 답변 반환 함수 정의
def gen(x, model, tokenizer):
    q = prompt % (x,)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=512,
        early_stopping=True,
        do_sample=False,
    )
    return tokenizer.decode(gened[0]).replace(q, "")