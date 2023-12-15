import pandas as pd


import torch

from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(100)

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('train.csv',encoding='utf-8')
print(train_df.head())

print("Number of records: ", train_df.shape[0])

valid_df = pd.read_csv('valid.csv',encoding='utf-8')
print(valid_df.head())

print("Number of records: ", valid_df.shape[0])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
INPUT_MAX_LEN = 512 # Input length
OUT_MAX_LEN = 128 # Output Length
TRAIN_BATCH_SIZE = 8 # Training Batch Size
VALID_BATCH_SIZE = 2 # Validation Batch Size
EPOCHS = 4 # Number of Iteration

MODEL_NAME = "t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, model_max_length= INPUT_MAX_LEN)


print("eos_token: {} and id: {}".format(tokenizer.eos_token,
                   tokenizer.eos_token_id)) # End of token (eos_token)
print("unk_token: {} and id: {}".format(tokenizer.unk_token,
                   tokenizer.eos_token_id)) # Unknown token (unk_token)
print("pad_token: {} and id: {}".format(tokenizer.pad_token,
                 tokenizer.eos_token_id)) # Pad token (pad_token)


class T5Dataset:

    def __init__(self, code, comment):
        self.code = code
        self.comment = comment
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUT_MAX_LEN

    def __len__(self):
        return len(self.code)

    def __getitem__(self, item):
        code = str(self.code[item])
        code = " ".join(code.split())

        comment = str(self.comment[item])
        comment = " ".join(comment.split())
        
        
        inputs_encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.input_max_len,
            padding = 'max_length',
            truncation='only_first',
            return_attention_mask=True,
            return_tensors="pt"
        )
        

        output_encoding = self.tokenizer(
            comment,
            None,
            add_special_tokens=True,
            max_length=self.out_max_len,
            padding = 'max_length',
            truncation= True,
            return_attention_mask=True,
            return_tensors="pt"
        )


        inputs_ids = inputs_encoding["input_ids"].flatten()
        attention_mask = inputs_encoding["attention_mask"].flatten()
        labels = output_encoding["input_ids"]

        labels[labels == 0] = -100  # As per T5 Documentation

        labels = labels.flatten()

        out = {
            "code": code,
            "comment": comment,
            "inputs_ids": inputs_ids,
            "attention_mask": attention_mask,
            "targets": labels
        }


        return out 



class T5DatasetModule(pl.LightningDataModule):

    def __init__(self, df_train, df_valid):
        super().__init__()
        self.df_train = df_train
        self.df_valid = df_valid
        self.tokenizer = tokenizer
        self.input_max_len = INPUT_MAX_LEN
        self.out_max_len = OUT_MAX_LEN


    def setup(self, stage=None):

        self.train_dataset = T5Dataset(
        code=self.df_train.code.values,
        comment=self.df_train.comment.values
        )

        self.valid_dataset = T5Dataset(
        code=self.df_valid.code.values,
        comment=self.df_valid.comment.values
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
         self.train_dataset,
         batch_size= TRAIN_BATCH_SIZE,
         shuffle=True, 
         num_workers=4
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
         self.valid_dataset,
         batch_size= VALID_BATCH_SIZE,
         num_workers=1
        )


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


def run():
    

    
    df_train = train_df.fillna("none")
    df_valid = valid_df.fillna("none")
    
    df_train['code'] = df_train['code'].apply(lambda x: " ".join(x.split()))
    df_valid['code'] = df_valid['code'].apply(lambda x: " ".join(x.split()))
    
    df_train['comment'] = df_train['comment'].apply(lambda x: " ".join(x.split()))
    df_valid['comment'] = df_valid['comment'].apply(lambda x: " ".join(x.split()))

   
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    dataModule = T5DatasetModule(df_train, df_valid)
    dataModule.setup()

    device = DEVICE
    models = T5Model()
    models.to(device)

    checkpoint_callback  = ModelCheckpoint(
        dirpath="./Output",
        filename="best_checkpoint",
        save_top_k=2,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        callbacks = checkpoint_callback,
        max_epochs= EPOCHS,
        accelerator="gpu",
	devices=2
    )

    trainer.fit(models, dataModule)

run()
