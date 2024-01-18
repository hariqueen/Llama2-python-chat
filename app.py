import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import re

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

st.set_page_config(page_title="My Llama2 Chatbot")

# 세션 상태 초기화 호출
initialize_session_state()

st.title(":koala: Llama2 Chatbot")
st.title(':blue_book: 파이썬 코드를 알려드릴게요.')

st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

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

base_model_path = "TinyPixel/CodeLlama-7B-Python-bf16-sharded"
peft_model_path = "hariqueen/myllama2"

@st.cache_resource
def load_model(base_model_path, peft_model_path):
    model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_path)
    return model

@st.cache_data
def load_tokenizer(base_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return tokenizer

model = load_model(base_model_path, peft_model_path)
tokenizer = load_tokenizer(base_model_path)

# 기존의 prompt 및 gen 함수 정의
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: %s ### Response: "

def gen(x):
    q = prompt % (x,)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ), 
        max_new_tokens=200,
        early_stopping=True,
        do_sample=False,
    )
    return tokenizer.decode(gened[0]).replace(q, "")

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요:")

# 사용자가 입력한 경우 응답 생성
if user_input:
    with st.spinner("Generating response..."):
        response = gen(user_input)
        response = re.sub(r'</?s>', '', response)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "chatbot", "content": response})

# 대화 이력 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "chatbot" else None):
        st.write(message["content"])