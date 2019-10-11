Data and preprocessing can be found: https://www.dropbox.com/s/vuhv2a7kgbjsi0s/data.zip?dl=0

This project is built on top of OpenNMT: https://github.com/OpenNMT/OpenNMT-py

To preprocess the data:

```bash
dataset=data_name
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```

To train the model:

```bash
dataset=data_name
model_name=model_name
python  train.py -data data/${dataset}/${dataset} \
                 -save_model experiments/${dataset}_${model_name} \
                 -gpu_ranks 0 -save_checkpoint_steps 10000  -keep_checkpoint 50 \
                 -train_steps 400000 -valid_steps 10000 -report_every 1000 -param_init 0  -param_init_glorot \
                 -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                 -learning_rate 2 -label_smoothing 0.0 \
                 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                 -dropout 0.1 -position_encoding -share_embeddings \
                 -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                 -heads 8 -transformer_ff 2048 -n_latent 1 -tensorboard -tensorboard_log_dir runs/${dataset}_${model_name}
```

To test the output results:

```bash
dataset=data_name
python parse/parse_output.py -input_file model_output_file -target_file data/${dataset}/tgt-test.txt -beam_size 10
```
