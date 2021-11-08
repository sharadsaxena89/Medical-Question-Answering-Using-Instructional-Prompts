The Preprocessing Steps will process the extracted data from 'Differential Diagnosis Primary Care', chunk it into smaller tables, generate question from question templates and segregate the dataset into train, test and dev.

The Singletask experiments code will train a model on single question template or task. Testing needs to be done 22 times for each template to get baseline. Change the filename and modelname in the file accordingly.

The Multitask experiments code will train a model on 17 question template or task. Testing will be done on the 17 tasks seen by the model. One set of files is wtihout prompt and other is with prompt. Do this for 3 train-test split. Total of 2x3x17 experiments. Change the trainining data, filename and modelname in the files accordingly.

For Crosstask experiments, we train on 17, and test on remaining 5 experiments. Total of 2x3x5 experiments. The paraphrasing experiments use the same code template as the Crosstask experiments.


