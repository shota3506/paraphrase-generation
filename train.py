import os
import argparse
import logging
import sentencepiece as spm
from tqdm import tqdm
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from criterion import LmCrossEntropyLoss, LabelSmoothedLmCrossEntropyLoss
from dataset import ParaphraseDataset, PAD_INDEX, UNK_INDEX, BOS_INDEX, EOS_INDEX

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--train_source_file", type=str, required=True)
parser.add_argument("--train_target_file", type=str, required=True)
parser.add_argument("--valid_source_file", type=str, required=True)
parser.add_argument("--valid_target_file", type=str, required=True)
parser.add_argument("--spm_file", type=str, required=True)
# Model
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--num_encoder_layers", type=int, default=6)
parser.add_argument("--num_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=512)
parser.add_argument("--dropout", type=float, default=.1)
parser.add_argument("--label_smoothing", type=float, default=.1)
parser.add_argument("--warmup_step", type=int, default=4000)
# Optim
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--checkpoint_file", type=str, default="model.pth")
parser.add_argument("--log_file", type=str, default="train.log")

args = parser.parse_args()


def main() -> None:
    logger = logging.getLogger(__name__)
    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler2 = logging.FileHandler(filename=args.log_file, mode='w')
    handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
    handler2.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    vocabulary_size = len(spm.SentencePieceProcessor(model_file=args.spm_file))
    train_dataset = ParaphraseDataset(args.train_source_file, args.train_target_file, tokenizer=spm.SentencePieceProcessor(model_file=args.spm_file).encode)
    valid_dataset = ParaphraseDataset(args.valid_source_file, args.valid_target_file, tokenizer=spm.SentencePieceProcessor(model_file=args.spm_file).encode)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, drop_last=True)

    ########## Self-Attention Network Encoder Decoder ##########
    from models.transformer import Transformer
    model = Transformer(
        num_embeddings=vocabulary_size,
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers, 
        dim_feedforward=args.dim_feedforward, 
        dropout=args.dropout).to(device)
    
    criterion = LabelSmoothedLmCrossEntropyLoss(PAD_INDEX, label_smoothing=args.label_smoothing, reduction='batchmean')

    lr_lambda = lambda step: model.d_model**(-0.5) * min((step+1)**(-0.5), (step+1) * args.warmup_step**(-1.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-09)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info('Start training')
    for epoch in range(args.num_epochs):
        train_loss, valid_loss = 0., 0.
        pbar = tqdm(train_loader)
        pbar.set_description("[Epoch %d/%d]" % (epoch, args.num_epochs))

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
            if itr % args.print_every == 0:
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
                
        logger.info('[Epoch %d] Training   loss: %.2f' % (epoch, train_loss))
        logger.info('           Validation loss: %.2f' % valid_loss)

        torch.save(model.state_dict(), args.checkpoint_file)
        
if __name__ == "__main__":
    main()

