import torch
import pandas as pd
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
from bert_model import * 

# 데이터 로드 및 분할
df = pd.read_csv('/dataset/random_data.csv')
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

# 설정값
max_len = 256
batch_size = 8
model_name = 'skt/kobert-base-v1'

# 토크나이저 및 데이터셋 초기화
tokenizer = KoBERTTokenizer.from_pretrained(model_name)
train_dataset = KoreanQuestionsDataset(df_train.Question.to_numpy(), df_train.label.to_numpy(), tokenizer, max_len)
val_dataset = KoreanQuestionsDataset(df_val.Question.to_numpy(), df_val.label.to_numpy(), tokenizer, max_len)

# 데이터 로더 설정
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))

# 모델 및 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# 학습 실행
for epoch in range(3):  # 에포크 수는 필요에 따라 조정
    train_loss = train(model, train_data_loader, optimizer, device)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.5f}')
