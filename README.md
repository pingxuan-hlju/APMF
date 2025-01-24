# APMF

## Introduction 
The project  is an implementation of a multi-granularity transformer contrastive learning and feature reconstruction for prediction of disease-related miRNAs

---

## Catalogs  
- **/Model**: Contains the code implementation of APMF algorithm.
- **dataloader.py**: Processes the miRNA, disease, and lncRNA similarities, associations, embeddings, and adjacency matrices.
- **model.py**: Defines the model.
- **main.py**: Trains the model.
---

## Environment 
The APMF code has been implemented and tested in the following development environment: 
 - Python == 3.7.1
 - PyTorch == 1.13.1
 - NumPy == 1.21.5
 - Matplotlib == 3.5.3

## Dataset 
In this work, the experimental data was extracted from the Human MicroRNA Disease Database (HMDD v4.0), which contains 1,245 miRNAs, 2,077 diseases, miRNA functional similarities, and a total of 23,337 miRNA-disease associations. The semantic similarities between diseases was calculated according to Medical Subject Headings (MeSH). The disease-lncRNA associations and miRNA-lncRNA interactions were obtained from the lncRNASNP data. 1,438 miRNA-lncRNA interactions, 320 lncRNA-disease interactions, and 557 lncRNAs were obtained. The dataset comes from https://github.com/AntarcticLu/MicRNA_Disease_Dataset.

## How to Run the Code  
1. **Data preprocessing**: Constructs the adjacency matrices, embeddings, and other inputs for training the model.  
    ```bash
    python dataloader.py
    ```  

2. **Train and test the model**.  
    ```bash
    python main.py
    ```  
