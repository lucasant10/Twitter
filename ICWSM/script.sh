#!/bin/bash


# sample <= 2000
sample=2000
# cbow_s100.txt, glove_s100.txt, skip_s100.txt, cbow_s300.txt, glove_s300.txt
embedding=cbow_s300.txt
# word-embeddings lengh: 100,200,300
dimension=300
# lstm.py, cnn.py
classification=cnn.py
# model_cnn, model_lstm
model_name=model_cnn
# dict_cnn, dict_lstm
dict=dict_cnn
# random, few_months, few_parls
dispersion=random
# word2vec, random
w_init=word2vec
# categorical_crossentropy, adam
loss=categorical_crossentropy

epoch=20
batch=30
max_len=18
echo "=============== Training Neural Network ============="
python $classification -f $embedding -d $dimension \
        --loss $loss --initialize-weights $w_init --learn-embeddings \
        --epochs $epoch --batch-size $batch \
        --dict_name $dict --model_name $model_name \
        --sample $sample --dispersion $dispersion
    
echo "=============== Trained NN Validation ============="
python validation.py -f $embedding -m $model_name -d $dict -l $max_len
    