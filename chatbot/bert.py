import streamlit as st
from transformers import BertForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer
import torch

# Bert 모델 불러오기
@st.cache_resource
def load_bert_model(bert_base_model, bert_model_path):
    bert_model = BertForSequenceClassification.from_pretrained(bert_base_model, num_labels=2)
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=torch.device('cuda')))
    return bert_model

# KoBERT 토크나이저 불러오기
@st.cache_data
def load_bert_tokenizer(bert_base_model):
    bert_tokenizer = KoBERTTokenizer.from_pretrained(bert_base_model)
    return bert_tokenizer

# 파이썬 관련 여부 확인 함수 정의
def check_answer(bert_model, bert_tokenizer, user_input):
    
    # 질문을 모델이 이해할 수 있는 형식으로 변환
    inputs = bert_tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    # 모델로 질문 추론
    bert_model.eval()           # 모델을 평가 모드로 설정
    with torch.no_grad():       # 그래디언트 계산을 하지 않음
        outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        prediction = torch.argmax(outputs.logits, dim=1)

    # 결과 반환
    predicted_label = prediction.cpu().numpy()[0]

    return predicted_label