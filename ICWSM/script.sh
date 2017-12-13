#!/bin/bash


# sample <= 2000
sample=100
# word-embeddings lengh: 100,200,300
dimension=100
# cbow_s100.txt, glove_s100.txt, skip_s100.txt, cbow_s300.txt, glove_s300.txt
embedding=glove_s100.txt
# lstm.py, cnn.py
classification=lstm.py
# model_cnn, model_lstm
model_name=model_lstm
# dict_cnn, dict_lstm
dict=dict_lstm
# random, few_month, few_parl
dispersion=random
# word2vec, random
w_init=word2vec
# categorical_crossentropy, adam
loss=categorical_crossentropy

epoch=20
batch=30
max_len=16
echo "=============== Training Neural Network ============="
# python $classification -f $embedding -d $dimension \
# 		--loss $loss --initialize-weights $w_init --learn-embeddings \
# 		--epochs $epoch --batch-size $batch \
#         --dict_name $dict --model_name $model_name \
#         --sample $sample --dispersion $dispersion

echo "=============== Trained NN Validation ============="
python validation.py -f $embedding -m $model_name -d $dict -l $max_len
