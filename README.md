# Paraphrase Generation
A simple sequence-to-sequence  paraphrase generator using Transformer.

## Usage instructions
### Train parpahrase generation model
```
python train.py \
--train_source_file <file_path> \
--train_target_file <file_path> \
--valid_source_file <file_path> \
--valid_target_file <file_path> \
--spm_file <file_path>
```

- ```spm_file``` SentencePiece model file

### Generate paraphrase
```
python generate.py \
--input_file <file_path> \
--output_file <file_path> \
--spm_file <file_path> \
--search_width 3
```
