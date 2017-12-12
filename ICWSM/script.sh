#!/bin/bash

input_dir=../../sample-data/
input_vec=../../sample-data/
output_dir=../../output/
model_dir=${output_dir}model/
mkdir -p $output_dir/model 

# sample <= 2000
sample = 100
# word-embeddings lengh: 100,200,300
dimension = 100
# cbow_s100.txt, glove_s100.txt, skip_s100.txt, cbow_s300.txt, glove_s300.txt
embedding = cbow_s100.txt
# lstm.py, cnn.py
classification = lstm.py
# model_cnn, model_lstm
model_name = model_lstm
# dict_cnn, dict_lstm
dict = dict_lstm
# uniform, one_month, uniform_parl, few_parl
dispersion = uniform
# word2vec, random
w_init = word2vec
# categorical_crossentropy, adam
loss = categorical_crossentropy

epoch = 20
batch = 30
max_len = 18
echo "=============== Training Neural Network ============="
python $classification -f $embedding -d $dimension \
		--loss $loss --initialize-weights $w_init --learn-embeddings \
		--epochs $epoch --batch-size $batch --dict_name $dict \ 
		--model_name $model_name --sample 100

echo "=============== Trained NN Validation ============="
python validation.py -f $embedding -m $model_name -d $dict -l $max_len
