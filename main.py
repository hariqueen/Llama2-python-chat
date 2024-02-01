import streamlit as st
import random
import re
from chatbot.bert import load_bert_model, load_bert_tokenizer, check_answer
from chatbot.codellama import load_model, load_tokenizer, gen
from chatbot.path import bert_base_model, bert_model_path, base_model_path, peft_model_path, answer_path
from db.insert import insert_data

# 파이썬과 무관한 질문일 경우 반환할 답변 데이터셋
with open(answer_path, 'r', encoding='utf-8') as file:
    answers = file.read().split('\n')

####################### 메인 화면 세팅 #######################

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


####################### 사이드바 #######################

st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")


# "새로운 질문하기" 버튼
if st.sidebar.button("새로운 질문 하기➕"):

    # 현재 대화 이력이 존재하고, 세션이 복원된 상태가 아닌 경우
    if st.session_state.chat_history and not st.session_state.get('restored_session', False):
        
        # 사이드바 이력에 사용자의 첫 번째 질문 추가
        first_user_question = next((msg for msg in st.session_state.chat_history if msg["role"] == "user"), None)
        
        if first_user_question:
            st.session_state.sidebar_history.append(first_user_question)
        
        # 현재 대화 이력을 full_history에 저장
        st.session_state.full_history.append(st.session_state.chat_history.copy())
    
    # 새로운 세션 시작
    st.session_state.chat_history = []
    
    # 세션이 복원된 상태를 False로 설정
    st.session_state.restored_session = False


# 사이드바의 질문 이력 표시 및 해당 대화 내역 복구
for idx, question in enumerate(st.session_state.sidebar_history):
    if st.sidebar.button(f"{idx + 1}. {question['content']}"):
        
        # 선택된 세션의 대화 이력 로드
        st.session_state.chat_history = st.session_state.full_history[idx]
        
        # 세션이 복원된 상태를 True로 설정
        st.session_state.restored_session = True


####################### 모델 캐싱 #######################

# codellama 모델 및 토크나이저 불러오기
model = load_model(base_model_path, peft_model_path)
tokenizer = load_tokenizer(base_model_path)

# bert 모델 및 토크나이저 불러오기
bert_model = load_bert_model(bert_base_model, bert_model_path)
bert_tokenizer = load_bert_tokenizer(bert_base_model)


####################### 답변 반환 #######################
        
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
        if check_answer(bert_model, bert_tokenizer, user_input) == 0:
            response = random.choice(answers)
        
        # 파이썬과 관련된 질문
        else:
            response = gen(user_input, model, tokenizer)
            response = re.sub(r'</?s>', '', response)

    # DB에 데이터 추가    
    insert_data(user_input, response)

    new_response = {"role": "chatbot", "content": response}
    st.session_state.chat_history.append(new_response)

    # 스크립트를 다시 실행하여 화면 갱신
    st.rerun()


####################### 채팅 내역 #######################

# 세션 상태에 저장된 이전 메시지들 표시
if not user_input:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐨" if message["role"] == "chatbot" else None):
            st.write(message["content"])