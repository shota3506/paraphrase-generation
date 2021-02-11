import os
import argparse
import sentencepiece as spm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from search import BeamSearch, DiverseBeamSearch, RandomSample
from dataset import ParaphraseDataset, PAD_INDEX, UNK_INDEX, BOS_INDEX, EOS_INDEX

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, default="output.txt")
parser.add_argument("--spm_file", type=str, required=True)
# Model
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--nhead", type=int, default=8)
parser.add_argument("--num_encoder_layers", type=int, default=6)
parser.add_argument("--num_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=512)
parser.add_argument("--dropout", type=float, default=.1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--checkpoint_file", type=str, default="model.pth")
# Search
parser.add_argument("--search_width", type=int, default=1)

args = parser.parse_args()

def main() -> None:
    subword_processor = spm.SentencePieceProcessor(model_file=args.spm_file)
    vocabulary_size = len(subword_processor)
    dataset = ParaphraseDataset(args.input_file, args.input_file, tokenizer=subword_processor.encode)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn, drop_last=False)

    searcher = BeamSearch(EOS_INDEX, beam_size=args.search_width)

    ########## Self-Attention Network Sequence to Sequence ##########
    from models.transformer import Transformer
        
    model = Transformer(
        num_embeddings=vocabulary_size,
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers, 
        dim_feedforward=args.dim_feedforward, 
        dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint_file, map_location=device))
    model.eval()

    print('Generating paraphrase...')
    all_hypotheses = []
    with torch.no_grad():
        for srcs, _ in tqdm(loader):
            srcs = srcs.to(device)
            bsz = len(srcs)
            src_key_padding_mask = (srcs == PAD_INDEX)
            memory_key_padding_mask = src_key_padding_mask

            memory = model.encode(srcs, src_key_padding_mask=src_key_padding_mask)
            start_predictions = torch.zeros(bsz, device=device).fill_(BOS_INDEX).long()
            start_state = {'memory': memory.permute(1, 0, 2), 'prev_output_tokens': None, 'memory_key_padding_mask': memory_key_padding_mask}
            predictions, log_probabilities = searcher.search(start_predictions, start_state, model.step)
    
            for preds in predictions:
                tokens = preds[0]
                tokens = tokens[tokens != EOS_INDEX].tolist()
                all_hypotheses.append(subword_processor.decode(tokens))
    print('Done')

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(all_hypotheses))

if __name__ == "__main__":
    main()
