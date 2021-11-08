import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Read the data
filename = '/home/ssaxen18/Thesis/prompt_tuning/1_train_file.tsv'
df = pd.read_csv(filename, delimiter='\t')

# Extract the contexts, questions and answers
train_contexts=[]
train_questions=[] 
train_answers=[]
for row_id in range(0,len(df)):
  train_contexts.append(df.iloc[row_id,1])
  train_questions.append(df.iloc[row_id,2])
  answer = {"answer_start": int(df.iloc[row_id,4]), "answer_end": int(df.iloc[row_id,5]),"text": df.iloc[row_id,3]} 
  train_answers.append(answer)
  
# Encode the data
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)

# Convert start, end position to token positions
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)

# Define class and override methods so that tensor parameters can be retrieved
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Cast the dataset to above class
train_dataset = SquadDataset(train_encodings)

# Load the model
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(4):
    total_train_loss = 0
    print('Running epoch: ', (epoch+1))
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        total_train_loss += outputs.loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
# Set parameter to evaluation
# model.eval()

model.save_pretrained('/home/ssaxen18/Thesis/prompt_tuning/model_1_train')
print('Model saved')
