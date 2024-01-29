import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BertForSequenceClassification
from peft import PeftModel
import re
import mysql.connector
from mysql.connector import Error
import datetime as dt
import torch
from kobert_tokenizer import KoBERTTokenizer
import random

########### DB 세팅 ###############

x = dt.datetime.now()
date = x.strftime("%Y-%m-%d")
time = x.strftime("%H:%M:%S")

# DB 연결 후 데이터 저장
def db_connect():
    try:
        # DB 연결
        conn = mysql.connector.connect(host = '127.0.0.1', database = 'testdb', user = 'coco', password = 'big5')
        if conn.is_connected():

            # 커서 생성
            cursor = conn.cursor()
            
            # 데이터 저장
            query = '''INSERT INTO test(question, answer, date, time) VALUES (%s, %s, %s, %s);'''
            input = (user_input, response, date, time)
            cursor.execute(query, input)
            
            # 작업 정상 처리
            conn.commit()

    # 에러 출력
    except Error as e :
        print('DB 에러 발생', e)

    # DB 연결 후 데이터 저장이 완료되면 커서와 커넥션 종료
    finally:
        cursor.close()
        conn.close()



######## 무관한 답변 세팅 (kobert)########
def check_answer(user_input):

    # 같은 모델 구조의 인스턴스 생성
    model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=2)

    model_path = '/home/user/cocobot/python_question_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

    # 토크나이저 초기화
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

    max_len = 512

    # 질문을 모델이 이해할 수 있는 형식으로 변환
    inputs = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    # 모델로 질문 추론
    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산을 하지 않음
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        prediction = torch.argmax(outputs.logits, dim=1)

    # 결과 반환
    predicted_label = prediction.cpu().numpy()[0]

    return predicted_label

# if 무관한 질문일 경우 답변셋
with open('./answer.txt', 'r', encoding='utf-8') as file:
    answers = file.read().split('\n')

############# page 세팅 시작 ################

st.set_page_config(page_title="My Llama2 Chatbot")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sidebar_history" not in st.session_state:
    st.session_state.sidebar_history = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []
if "session_active" not in st.session_state:
    st.session_state.session_active = False

st.title(":koala: Coala")
st.title(':blue_book: 파이썬에 대해 언제든지 물어보세요.')


##### 사이드바 #######
st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

# "새로운 질문하기" 버튼 로직
if st.sidebar.button("새로운 질문하기➕"):

    # 현재 대화 이력이 존재하고, 세션이 복원된 상태가 아닌 경우 처리
    if st.session_state.chat_history and not st.session_state.get('restored_session', False):
        
        # 첫 번째 사용자 질문 찾기
        first_user_question = next((msg for msg in st.session_state.chat_history if msg["role"] == "user"), None)
        
        if first_user_question:
            
            # 사이드바 이력에 첫 번째 사용자 질문 추가
            st.session_state.sidebar_history.append(first_user_question)
        
        # 현재 대화 이력을 full_history에 저장
        st.session_state.full_history.append(st.session_state.chat_history.copy())
    
    # 새로운 세션 시작
    st.session_state.chat_history = []
    
    # 세션이 복원된 상태를 False로 설정
    st.session_state.restored_session = False

# 사이드바의 질문 이력 표시 및 복구
for idx, question in enumerate(st.session_state.sidebar_history):
    if st.sidebar.button(f"{idx + 1}. {question['content']}"):
        
        # 선택된 세션의 대화 이력 로드
        st.session_state.chat_history = st.session_state.full_history[idx]
        
        # 세션이 복원된 상태를 True로 설정
        st.session_state.restored_session = True

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
peft_model_path = "kgyalice/llama2_peft"

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
        ).to('cuda'),
        max_new_tokens=512,
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

    # 대화 이력 표시 (사용자 질문 포함)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])

    # 응답 생성 시작 전 스피너 구현
    with st.spinner("답변 중이에요.."):

        # 파이썬과 무관한 질문
        if check_answer(user_input) == 0:
            response = random.choice(answers)

        # 파이썬과 관련된 질문
        else:
            response = gen(user_input)
            response = re.sub(r'</?s>', '', response)

    db_connect()

    new_response = {"role": "chatbot", "content": response}
    st.session_state.chat_history.append(new_response)

    # 스크립트를 다시 실행하여 화면 갱신
    st.rerun()

# 세션 상태에 저장된 이전 메시지들 표시
if not user_input:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])