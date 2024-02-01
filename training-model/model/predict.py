import torch

__all__ : ["predict"]

def predict(model, tokenizer, sentence, max_len, device):
    model.eval()
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
        prediction = torch.argmax(outputs.logits, dim=1)
    return prediction.cpu().numpy()[0]
