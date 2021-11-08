import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
	print('There are %d GPU(s) available.' % torch.cuda.device_count())

	print('We will use the GPU:', torch.cuda.get_device_name(0))


from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("/home/ssaxen18/Thesis/prompt_tuning/model_1_train/")
model.eval()
model.to(device)


from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Read the data
filename = '/home/ssaxen18/Thesis/prompt_tuning/1_test_file.tsv'
df = pd.read_csv(filename, delimiter='\t')

# Extract the contexts, questions and answers
test_contexts=[]
test_questions=[] 
test_answers=[]
for row_id in range(0,len(df)):
  test_contexts.append(df.iloc[row_id,1])
  test_questions.append(df.iloc[row_id,2])
  answer = {"answer_start": int(df.iloc[row_id,4]), "answer_end": int(df.iloc[row_id,5]),"text": df.iloc[row_id,3]} 
  test_answers.append(answer)
  
# Encode the data
test_encodings = tokenizer(test_contexts, test_questions, max_length=512, truncation=True, padding=True)

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

add_token_positions(test_encodings, test_answers)

# Define class and override methods so that tensor parameters can be retrieved
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Cast the dataset to above class
test_dataset = SquadDataset(test_encodings)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Tracking variables 
total_test_accuracy = 0
total_test_loss = 0

pred_start, pred_end, true_start, true_end = [], [], [], []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
    
    loss = outputs.loss
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits 
    
    # Accumulate the validation loss.
    total_test_loss += loss.item()
    
    # Move logits and labels to CPU
    start_logits = start_logits.detach().cpu().numpy()
    end_logits = end_logits.detach().cpu().numpy()    
    
    # Move the correct start and end positions back to the CPU.
    start_positions = start_positions.to('cpu').numpy()
    end_positions = end_positions.to('cpu').numpy()
    
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = np.argmax(start_logits, axis=1)
    answer_end = np.argmax(end_logits, axis=1)

    # Store predictions and true labels
    pred_start.append(answer_start)
    pred_end.append(answer_end)
    true_start.append(start_positions)
    true_end.append(end_positions)

# Combine the results across the batches.
pred_start = np.concatenate(pred_start, axis=0)
pred_end = np.concatenate(pred_end, axis=0)
true_start = np.concatenate(true_start, axis=0)
true_end = np.concatenate(true_end, axis=0)
    
# Count up the number of start index predictions and end index predictions 
# which match the correct indeces.
num_start_correct = np.sum(pred_start == true_start)
num_end_correct = np.sum(pred_end == true_end)

total_correct = num_start_correct + num_end_correct
total_indeces = len(true_start) + len(true_end)

# Report the final accuracy for this validation run.
avg_test_accuracy = float(total_correct) / float(total_indeces)
print("  Accuracy: {0:.2f}".format(avg_test_accuracy))

# Calculate the average loss over all of the batches.
avg_test_loss = total_test_loss / len(test_loader)

# print("  Test Loss: {0:.2f}".format(avg_test_loss))


# The final F1 score for each sample.
f1s = []

# For each test sample...
for i in range(0, len(pred_start)):

    # Expand the start and end indeces into sequences of indeces stored as sets.
    # For example, if pred_start = 137 and pred_end = 140, then
    #   pred_span = {137, 138, 139, 140}
    pred_span = set(range(pred_start[i], pred_end[i] + 1))


    # f1_options = []

    # For each of the three possible answers...
    #for j in range (0, len(start_positions[i])):
    
        # Expand this answer into a range, as above.
    true_span = set(range(true_start[i], true_end[i] + 1))    

    # Use the `intersection` function from Python `set` to get the set of 
    # indeces occurring in both spans. Take the length of this resulting set
    # as the number of overlapping indeces between the two spans.
    num_same = len(pred_span.intersection(true_span))    

    # If there's no overlap, then the F1 score is 0 for this sample.
    if num_same == 0:
        f1 = 0
        continue

    # Precision - How many tokens overlap relative to the total number of tokens
    #             in the predicted span? If the model predicts too large of a 
    #             span, it has bad precision.      
    precision = float(num_same) / float(len(pred_span))

    # Recall - How many of the correct tokens made it into the predicted span?
    #          A model could have perfect recall if it just predicted the entire
    #          paragraph as the answer :).    
    recall = float(num_same) / float(len(true_span))

    # F1 - Does the model have both good precision and good recall?
    f1 = (2 * precision * recall) / (precision + recall)

    # Store the score.
    # f1_options.append(f1)

        # ^^^ Continue looping through possible answers ^^^

    # Take the highest of the three F1 scores as our score for this sample.
    f1s.append(f1)

    # ^^^ Continue looping through test samples ^^^


print('Average F1 Score: {:.3f}'.format(np.mean(f1s)))

