import time
import os

start_time = time.time()
command = 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
os.system(command)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The cache_clearning took {elapsed_time} seconds to execute.\n")

print("bert test starts...")
start_time = time.time()
import torch
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The Torch_framework_loading took {elapsed_time} seconds to execute.\n")

from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # This model predicts a sentiment score from 1-5

model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
# Save the BERT model
model_save_path = "org_bert_model.bin"
torch.save(model.state_dict(), model_save_path)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# Save the tokenizer
tokenizer_save_path = "org_bert_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)

# Define the review text
review_text = "The movie was fantastic! I really enjoyed it. I can't wait to watch it again in the next few days."

start_time = time.perf_counter()
# Tokenize the review text and obtain the input tensors
inputs = tokenizer.encode_plus(
    review_text,
    add_special_tokens=True,
    max_length=256,
    return_tensors="pt",
    padding='max_length',  # This replaces pad_to_max_length=True
    truncation=True
)


input_ids = inputs["input_ids"]
# Save the input_ids
input_ids_save_path = "org_input_ids.bin"
torch.save(input_ids, input_ids_save_path)

attention_mask = inputs["attention_mask"]
# Save the attention_mask
attention_mask_save_path = "org_attention_mask.bin"
torch.save(attention_mask, attention_mask_save_path)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The input_data_generating took {elapsed_time} seconds to execute.\n")


print("checking if cuda is available...")
start_time = time.perf_counter()
# Check for GPU availability and move the tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("finish checking cuda...")
print("device:", device)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The CUDA_device_setup took {elapsed_time} seconds to execute.\n")

start_time = time.perf_counter()
print("\nPutting input_ids.to(device)...")
input_ids = input_ids.to(device)
print("\nPutting attention_mask.to(device)...")
attention_mask = attention_mask.to(device)
print("\nPutting model.to(device)...")
#print("model dict:", model.__dict__)
model = model.to(device)
print("\nPrepare model to evaluation mode...")
# Perform inference
model.eval()
print("\nSetting torch to no_grad mode...")
with torch.no_grad():
    print("\nGetting the model inference result...")
    logits = model(input_ids, attention_mask=attention_mask)[0]

print("\n Getting the predicted sentiment...")
# Get the predicted sentiment
predicted_class = torch.argmax(logits, dim=1).item()
print(f"Predicted sentiment score: {predicted_class}")
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The total_model_inferencing took {elapsed_time} seconds to execute.\n")

import ctypes

# Load the shared library
cuda_interceptor = ctypes.CDLL('./my_cuda_wrappers.so')

# Call the function signal_sending()
cuda_interceptor.signal_sending()

print("clearing cache...")
command = 'sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
#os.system(command)




