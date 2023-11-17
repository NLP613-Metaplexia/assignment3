# Assignment 3

## Team Name: Metaplexia
### Members: 


| Sl. No | Name                | Reg No.  | Task | Percentage |
|--------|---------------------|----------|------|-----|
| 1      | Ritesh Patidar      | 22210034 | Pre-processing of the Dataset | 16 |
| 2      | Siddhesh Dosi       | 22210045 | Pretraining pipeline of the BERT model | 16 |
| 3      | Anupam Sharma       | 22210006 | Documentation of the Finetuning notebook |16|
| 4      | Kowsik Nandagopan D | 22250016 | Coding of Classification and QA Task |16|
| 5      | Ankit Yadav         | 22270001 | Computation of the perplexity metric |16|
| 6      | Sai Krishna Avula   | 22210036 | Documentation of the pre-training |2|
| 7      | Hitesh lodwal       | 22210019 | Finetuning classification task and QA Task |16| 
| 8      | Ayush Shrivastava   | 22210010 | Reading QA Task documentation |2|

## Answers to the Question
### Pre-training Pipeline
Pre-training code and pipeline are provided in this repository under the name [bert_pre_train.ipynb](https://github.com/NLP613-Metaplexia/assignment3/blob/main/bert_pre_train.ipynb)
1. We selected [bert-base-uncased](https://huggingface.co/bert-base-uncased) from the 🤗 repository
2. The number of parameters in the _pre-trained_ model (BERT encoder architecture) is 109,514,298. In the paper, 110M parameters. The number we got through our experiment is approximately the same as defined in the paper.
3. We pretrained the model after re-initializing the weights and trained on the train split of the [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-raw-v1). We discovered that, the best hyper-parameters for the task is `3e-4` learning rate, batch size of `32` weight decay of `0.1`, and number of epochs as 10.
4. Perplexity of the model test split of the [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/viewer/wikitext-2-raw-v1) came around **5.02**. 
5. We have pushed the model to the 🤗 repository [temporary0-0name/run_opt](https://huggingface.co/temporary0-0name/run_opt)
---   

### Fine-tuning Pipeline
Fine-tuning code and pipeline are provided in this repository under the name [bert_fine_tune.ipynb](https://github.com/NLP613-Metaplexia/assignment3/blob/main/bert_fine_tune.ipynb)

6. (a) In the **Classification** task, the [SST2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2) dataset was fine-tuned on the pre-trained model in the previous step.
   (b) In the **Question-Answering** task [SQuAD](https://huggingface.co/datasets/squad_v2) dataset was finetuned on the pre-trained model in step 5
7. In the 20% test split (seed=1) of the dataset for both tasks, the model got the following metrics (NA means: not mentioned in the question to compute)

   
 | **Task/Metric**    | **Accuracy** | **Precision** | **Recall** | **F1** | **Exact Match** | **METEOR** | **BLEU** | **ROUGE** |
|--------------------|--------------|---------------|------------|--------|-----------------|------------|----------|-----------|
| Classification     | 0.557        | 0.557         | 1.0        | 0.557  | NA              | NA         | NA       | NA        |
| Question-Answering | NA           | NA            | NA         |        |                 |            |          |           |  


8. The number of parameters after finetuning the model is as follows,
   
|                        | **Number of Parameters** | **Reason** |
|------------------------|--------------------------|------------|
| BERT in Classification |                          |            |
| BERT for QA Task       |                          |            |

9. We have pushed the model to the 🤗 repository. For the **Classification** task, the model is available in [Hitesh1501/sst2](https://huggingface.co/Hitesh1501/sst2). Similarly, for the **Question-Answering** task the fine-tuned model is in
 [Hitesh1501/squad](https://huggingface.co/Hitesh1501/squad)
10. 
