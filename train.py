import os
from tqdm import tqdm
from statistics import mean
import sentencepiece as spm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from criterion import LmCrossEntropyLoss, LabelSmoothedLmCrossEntropyLoss
from dataset import ParaphraseDataset, PAD_INDEX, UNK_INDEX, BOS_INDEX, EOS_INDEX

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

def main() -> None:
    data_dir = "data/quora"
    train_src_file = os.path.join(data_dir, "train.src.txt")
    train_ref_file = os.path.join(data_dir, "train.ref.txt")
    valid_src_file = os.path.join(data_dir, "valid.src.txt")
    valid_ref_file = os.path.join(data_dir, "valid.ref.txt")
    
    spm_file = "data/quora/spm.model"
    vocabulary_size = len(spm.SentencePieceProcessor(model_file=spm_file))
    train_dataset = ParaphraseDataset(train_src_file, train_ref_file, tokenizer=spm.SentencePieceProcessor(model_file=spm_file).encode)
    valid_dataset = ParaphraseDataset(valid_src_file, valid_ref_file, tokenizer=spm.SentencePieceProcessor(model_file=spm_file).encode)
    batch_size = 128
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, drop_last=True)

    ########## Self-Attention Network Encoder Decoder ##########
    from models.transformer import Transformer

    d_model = 256
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1

    model = Transformer(
        num_embeddings=vocabulary_size,
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout).to(device)
    
    label_smoothing = 0.1
    warmup_step = 4000
    criterion = LabelSmoothedLmCrossEntropyLoss(PAD_INDEX, label_smoothing=label_smoothing, reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-09)
    lr_lambda = lambda step: model.d_model**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_step**(-1.5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    epochs = 100
    print_every = 100
    for epoch in range(epochs):
        train_loss, valid_loss = 0., 0.
        pbar = tqdm(train_loader)
        pbar.set_description("[Epoch %d/%d]" % (epoch, epochs))

        # Train
        model.train()
        for itr, (srcs, tgts) in enumerate(pbar):
            srcs, tgts = srcs.to(device), tgts.to(device)
            src_key_padding_mask = (srcs == PAD_INDEX)
            tgt_key_padding_mask = (tgts == PAD_INDEX)
            memory_key_padding_mask = src_key_padding_mask
            tgt_mask = model.generate_square_subsequent_mask(tgts.size(1)).to(device)

            output = model(srcs, tgts, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            loss = criterion(output[:, :-1, :], tgts[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
            if itr % print_every == 0:
                pbar.set_postfix(loss=train_loss / (itr + 1), lr=scheduler.get_last_lr()[0])
        train_loss /= len(train_loader)

        # Valid
        model.eval()
        with torch.no_grad():
            for (srcs, tgts) in valid_loader:
                srcs, tgts = srcs.to(device), tgts.to(device)
                src_key_padding_mask = (srcs == PAD_INDEX)
                tgt_key_padding_mask = (tgts == PAD_INDEX)
                memory_key_padding_mask = src_key_padding_mask
                tgt_mask = model.generate_square_subsequent_mask(tgts.size(1)).to(device)

                output = model(srcs, tgts, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, 
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
                loss = criterion(output[:, :-1, :], tgts[:, 1:])
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
                
        print('Training   loss: %.2f\nValidation loss: %.2f' % (train_loss, valid_loss))

        torch.save(model.state_dict(), "model.pth")
        
if __name__ == "__main__":
    main()

