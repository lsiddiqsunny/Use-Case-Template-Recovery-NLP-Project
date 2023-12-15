import pandas as pd


import torch

from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(100)

import warnings
warnings.filterwarnings("ignore")


test_df = pd.read_csv('test.csv',encoding='utf-8')
print(test_df.head())

print("Number of records: ", test_df.shape[0])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
INPUT_MAX_LEN = 512 # Input length
OUT_MAX_LEN = 128 # Output Length
TRAIN_BATCH_SIZE = 8 # Training Batch Size
VALID_BATCH_SIZE = 2 # Validation Batch Size
EPOCHS = 5 # Number of Iteration

MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length= INPUT_MAX_LEN)


class T5Model(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

        return output.loss, output.logits


    def training_step(self, batch, batch_idx):

        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["targets"]
        loss, outputs = self(input_ids, attention_mask, labels)

        
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["inputs_ids"]
        attention_mask = batch["attention_mask"]
        labels= batch["targets"]
        loss, outputs = self(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

train_model = T5Model.load_from_checkpoint("./Output/best_checkpoint.ckpt")

train_model.freeze()

def generate_question(code):
    
    inputs_encoding =  tokenizer(
        code,
        add_special_tokens=True,
        max_length= INPUT_MAX_LEN,
        padding = 'max_length',
        truncation='only_first',
        return_attention_mask=True,
        return_tensors="pt"
        )

    
    generate_ids = train_model.model.generate(
        input_ids = inputs_encoding["input_ids"],
        attention_mask = inputs_encoding["attention_mask"],
        max_length = INPUT_MAX_LEN,
        num_beams = 4,
        num_return_sequences = 1,
        no_repeat_ngram_size=2,
        early_stopping=True,
        )

    preds = [
        tokenizer.decode(gen_id,
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True)
        for gen_id in generate_ids
    ]

    return "".join(preds)


test_df['solution'] = test_df['code'].apply(lambda x: generate_question(x))

test_df.to_csv('test_output.csv',index=False)