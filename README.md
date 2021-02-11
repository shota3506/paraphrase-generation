# Paraphrase Generation
This is a PyTorch implementation of the simple sequence-to-sequence  paraphrase generator using Transformer (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

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
- ```train_source_file``` one-sentence-per-line raw corpus file for training.
- ```train_target_file``` one-sentence-per-line raw corpus file for training.
- ```valid_source_file``` one-sentence-per-line raw corpus file for validation.
- ```valid_target_file``` one-sentence-per-line raw corpus file for validation.
- ```spm_file``` SentencePiece model file.

### Generate paraphrase
```
python generate.py \
--input_file <file_path> \
--output_file <file_path> \
--spm_file <file_path> \
--search_width 3
```

- ```input_file``` one-sentence-per-line raw corpus file to paraphrase. 
- ```output_file``` output file
- ```search_width``` beam search width.

## Example
### [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
Download data from [here](https://drive.google.com/drive/folders/1d8bfnRvSW0nD-EMlScyY47vIw8RoQda0?usp=sharing).
Download trained model parameters and generated paraphrases from [here](https://drive.google.com/drive/folders/1MnyDvB9SQCkOLdjBjZqGU20RqzadDXaK?usp=sharing).
