# Assignment 3

## Team Name: Metaplexia
### Members: 


| Sl. No | Name                | Reg No.  | Task | Percentage |
|--------|---------------------|----------|------|-----|
| 1      | Ritesh Patidar      | 22210034 | Pre-processing of the Dataset | 12.5 |
| 2      | Siddhesh Dosi       | 22210045 | Pretraining pipeline of the BERT model |12.5|
| 3      | Anupam Sharma       | 22210006 | Documentation of the Finetuning notebook |12.5|
| 4      | Kowsik Nandagopan D | 22250016 | Coding of Classification and QA Task |12.5|
| 5      | Ankit Yadav         | 22270001 | Computation of the perplexity metric |12.5|
| 6      | Sai Krishna Avula   | 22210036 | Documentation of the pre-training |12.5|
| 7      | Hitesh lodwal       | 22210019 | Finetuning classification task and QA Task |12.5| 
| 8      | Ayush Shrivastava   | 22210010 | Reading QA Task documentation, verification and review of code |12.5|

## Answers to the Question
### Pre-training Pipeline
Pre-training code and pipeline are provided in this repository under the name [bert_pre_train.ipynb](https://github.com/NLP613-Metaplexia/assignment3/blob/main/bert_pre_train.ipynb)
1. We selected [bert-base-uncased](https://huggingface.co/bert-base-uncased) from the 🤗 repository
2. The number of parameters in the _pre-trained_ model (BERT encoder architecture) is 109,514,298. In the paper, 110M parameters. The number we got through our experiment is approximately the same as defined in the paper.
3. We pretrained the model after re-initializing the weights and trained on the train split of the [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-raw-v1). We discovered that the best hyper-parameters for the task is `3e-4` learning rate, batch size of `32` weight decay of `0.1`, and number of epochs as 10.
4. Perplexity of the model test split of the [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-raw-v1) came around **5.02**. We presume that the model has overfitted on the dataset; the test dataset is not able to genealise the understanding of the language model. 
5. We have pushed the model to the 🤗 repository [temporary0-0name/run_opt](https://huggingface.co/temporary0-0name/run_opt)
---   

| Epoch | Perplexity |
|-------|------------|
| 1     | 904.00     |
| 2     | 218.44     |
| 3     | 17.65      |
| 4     | 8.63       |
| 5     | 6.8        |
| 6     | 6.63       |
| 7     | 6.66       |
| 8     | 6.88       |
| 9     | 8.46       |
| 10    | 8.16       |

### Fine-tuning Pipeline
Fine-tuning code and pipeline are provided in this repository under the name [bert_fine_tune.ipynb](https://github.com/NLP613-Metaplexia/assignment3/blob/main/bert_fine_tune.ipynb)

6. (a) In the **Classification** task, the [SST2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2) dataset was fine-tuned on the pre-trained model in the previous step.

   (b) In the **Question-Answering** task [SQuAD](https://huggingface.co/datasets/squad_v2) dataset was finetuned on the pre-trained model in step 5

7. In the 20% test split (seed=1) of the dataset for both tasks, the model got the following metrics (NA means: not mentioned in the question to compute)

   
 | **Task/Metric**    | **Accuracy** | **Precision** | **Recall** | **F1** | **Exact Match** | **METEOR** | **BLEU** | **ROUGE** |
|--------------------|--------------|---------------|------------|--------|-----------------|------------|----------|-----------|
| Classification     | 0.557        | 0.557         | 1.0        | 0.557  | NA              | NA         | NA       | NA        |
| Question-Answering | NA           | NA            | NA         |   1.0     |     0.0            |    0.02        |     0.01     |      0.12     |  


8. The number of parameters after finetuning the model is as follows,
   
|                        | **Number of Parameters** | **Reason** |
|------------------------|--------------------------|------------|
| BERT in Classification | 109,483,778              |   Number of parameter are less than the pretrained model because in pretrained language model output layer is of vocab size here there is one pooling layer and classification layer.         |
| BERT for QA Task       | 108,893,186                  |   Number of parameter are less than the pretrained model because in pretrained language model output layer is of vocab size but here it lags pooling layer after 12th encoder. Additionally it has only classification head.          |

9. We have pushed the model to the 🤗 repository. For the **Classification** task, the model is available in [Hitesh1501/sst2](https://huggingface.co/Hitesh1501/sst2). Similarly, for the **Question-Answering** task the fine-tuned model is in
 [Hitesh1501/squad](https://huggingface.co/Hitesh1501/squad)
10. (a) The model gives **poor performace** because the pretaining task done for just five epochs which resulted in lack of language understanding by the BERT Model. We are receiving 5.02 perplexity because the model overfits on the pretraining dataset thus lacks the generalizability on the downstream task like Classification and Question-answering.
    
    (b)  The number of parameters in the _pre-trained_ model (BERT encoder architecture) is 109,514,298. In the paper, 110M parameters. The number we got through our experiment is approximately the same as defined in the paper. In Classification and QA the number of parameter are less than the pretrained model. In pretrained model the output head is of vocab size. In fine-tuning task the output head just contains the classification head , sometimes their might been a hidden layer between the 12th layer and the classification layer. More details are provided in answer corresponding to 2 and 8. 
