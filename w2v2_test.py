import transformers, torch

model = transformers.Wav2Vec2Model.from_pretrained('charsiu/en_w2v2_fc_10ms')

output = model(torch.zeros((1, 399))).last_hidden_state