import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import random

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

complete_train_contexts=[]
complete_train_questions=[] 
complete_train_answers=[]

print('Reading the files......')
list_of_files = [2,3,5,6,8,9,10,11,12,13,14,16,17,18,19,20,22]

prompt_filename = '/home/ssaxen18/Thesis/prompt_tuning/prompt.tsv'
df_prompt = pd.read_csv(prompt_filename, delimiter='\t')

directory = '/home/ssaxen18/Thesis/prompt_tuning/qa_dataset_v2/'
for file_id in list_of_files:
    print(file_id)
    prompt_id = file_id
    prompt = df_prompt.iloc[(prompt_id-1),1] +'Question: '+ df_prompt.iloc[(prompt_id-1),2] +'Answer: ' + df_prompt.iloc[(prompt_id-1),3]
    
    train_filename = directory + str(file_id) +'_train_file.tsv'
    
    df = pd.read_csv(train_filename, delimiter='\t')
    for row_id in range(0,len(df)):
      context = df.iloc[row_id,1] + prompt
      length_context = len((tokenizer(context))['input_ids'])
      length_question = len((tokenizer(df.iloc[row_id,2]))['input_ids'])
      if ((length_context+length_question)<512):
          complete_train_contexts.append(context)
          complete_train_questions.append(df.iloc[row_id,2])
          answer = {"answer_start": int(df.iloc[row_id,4]), "answer_end": int(df.iloc[row_id,5]),"text": df.iloc[row_id,3]} 
          complete_train_answers.append(answer)
      
print('Files read successfully')

# Shuffling the entries
list_of_entries = []
for i in range(0,len(complete_train_contexts)):
    list_of_entries.append([complete_train_contexts[i],complete_train_questions[i],complete_train_answers[i]])
random.shuffle(list_of_entries)
complete_train_contexts=[]
complete_train_questions=[] 
complete_train_answers=[]
for i in range(0,len(list_of_entries)):
    complete_train_contexts.append(list_of_entries[i][0])
    complete_train_questions.append(list_of_entries[i][1])
    complete_train_answers.append(list_of_entries[i][2])
# End of shuffle code

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


# Define class and override methods so that tensor parameters can be retrieved
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


# Load the model
from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(4):
    total_train_loss = 0
    print('Running epoch: ', (epoch+1))
    final_start_index = len(complete_train_contexts) - 2000
    for start_index in range(0,final_start_index,2000):
        end_index = min( (start_index+2000), len(complete_train_contexts) )
        train_questions =  complete_train_questions[start_index:end_index]
        train_contexts = complete_train_contexts[start_index:end_index]
        train_answers = complete_train_answers[start_index:end_index]
        print("Tokenizing entries: {:} to {:}".format(start_index,end_index))
        train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        add_token_positions(train_encodings, train_answers)

        train_dataset = SquadDataset(train_encodings)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
             
            
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

model.save_pretrained('/home/ssaxen18/Thesis/prompt_tuning/multitask_model_prompt')
print('Model saved')
