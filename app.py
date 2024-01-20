
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import re

st.set_page_config(page_title="My Llama2 Chatbot")

# 세션 상태에 대화 이력과 사이드바 질문 이력을 저장할 리스트 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sidebar_history" not in st.session_state:
    st.session_state.sidebar_history = []

st.title(":koala: Llama2 Chatbot")
st.title(':blue_book: 파이썬 코드를 알려드릴게요.')


##### 사이드바 #######
st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

if st.sidebar.button("질문 이력 삭제"):
    # 메인 화면의 대화 이력 초기화
    st.session_state.chat_history = []
    st.sidebar.success("질문 이력이 삭제되었습니다.")

with st.sidebar:
    for message in st.session_state.sidebar_history:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
                button = st.button(message['content'])
        if button and message["role"] == "chatbot":
            with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
                st.write(message["content"])

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
user_input = st.chat_input("질문은 여기에 입력해 주세요.")

# 사용자가 입력한 경우
if user_input:
    # 먼저 사용자의 질문을 채팅 창과 사이드바 이력에 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.sidebar_history.append({"role": "user", "content": user_input})

    # 대화 이력 표시 (사용자 질문 포함)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])

    # 응답 생성 시작 전 "답변중이에요.." 메시지 표시
    with st.spinner("답변중이에요.."):
        # 응답 생성
        response = gen(user_input)
        response = re.sub(r'</?s>', '', response)

    # 생성된 응답을 세션 상태의 두 이력에 모두 추가
    st.session_state.chat_history.append({"role": "chatbot", "content": response})
    st.session_state.sidebar_history.append({"role": "chatbot", "content": response})

    # 스크립트를 다시 실행하여 화면 갱신
    st.experimental_rerun()

# 세션 상태에 저장된 이전 메시지들 표시
# (이 부분은 사용자 입력이 없을 때에만 실행됨)
if not user_input:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])