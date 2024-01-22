
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import re

st.set_page_config(page_title="My Llama2 Chatbot")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sidebar_history" not in st.session_state:
    st.session_state.sidebar_history = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []

st.title(":koala: Llama2 Chatbot")
st.title(':blue_book: 파이썬 코드를 알려드릴게요.')


##### 사이드바 #######
st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

if st.sidebar.button("새로운 질문하기➕"):
    if st.session_state.chat_history:
        st.session_state.full_history.append(st.session_state.chat_history.copy())
    st.session_state.chat_history = []
    # 첫 번째 사용자 질문 유지
    if st.session_state.sidebar_history and st.session_state.sidebar_history[0]["role"] == "user":
        first_user_question = st.session_state.sidebar_history[0]
        st.session_state.sidebar_history = [first_user_question]
    else:
        st.session_state.sidebar_history = []

# 사이드바의 첫 번째 질문 표시 및 이력 복구
with st.sidebar:
    if st.session_state.sidebar_history:
        first_question = st.session_state.sidebar_history[0]
        if first_question["role"] != "placeholder" and st.sidebar.button(first_question['content']):
            st.session_state.chat_history = st.session_state.full_history[-1].copy()


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

# 사용자가 새로운 질문을 입력한 경우
if user_input:
    new_message = {"role": "user", "content": user_input}
    st.session_state.chat_history.append(new_message)

    # sidebar_history에 요소가 있고 첫 번째 요소가 플레이스홀더인 경우 업데이트
    if st.session_state.sidebar_history and st.session_state.sidebar_history[0]["role"] == "placeholder":
        st.session_state.sidebar_history[0] = new_message
    elif not st.session_state.sidebar_history:
        st.session_state.sidebar_history.append(new_message)

    # 대화 이력 표시 (사용자 질문 포함)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])

    # 응답 생성 시작 전 스피너 구현
    with st.spinner("답변중이에요.."):
        response = gen(user_input)
        response = re.sub(r'</?s>', '', response)

    new_response = {"role": "chatbot", "content": response}
    st.session_state.chat_history.append(new_response)

    # 스크립트를 다시 실행하여 화면 갱신
    st.experimental_rerun()

# 세션 상태에 저장된 이전 메시지들 표시
if not user_input:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])