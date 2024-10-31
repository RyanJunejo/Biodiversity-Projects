# Advancing Biodiversity Conservation: Comparative Evaluation of Machine Learning Models for Species Classification

Hello! This is a collection of the work I did at my internship at the Vector Institute under the supervision of Graham Taylor.
This repository contains code, data, and documentation for our project on evaluating machine learning models for species classification, focusing on biodiversity conservation. This project compares various models to assess their effectiveness in classifying species, particularly rare and underrepresented ones. The project also proposes novel methodologies for model evaluation to improve species classification accuracy and robustness.
I strongly suggest you click on the powerpoint presentation with the same name to learn more!

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
  - [Hierarchical Vision Transformers (Hiera)](#hierarchical-vision-transformers-hiera)
  - [BioCLIP](#bioclip)
- [Methods](#methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Contributing](#contributing)


---

## Introduction
In this project, we explore machine learning (ML) approaches for species classification to support biodiversity conservation. Accurate species classification is essential for monitoring biodiversity, preserving rare species, and supporting conservation efforts worldwide. By comparing models such as **Hiera** and **BioCLIP**, we aim to improve classification accuracy and evaluate model effectiveness, particularly for rare or difficult-to-distinguish species. This project also proposes tailored metrics to assess model performance more effectively.

## Dataset Description
We use three key datasets in this project:
1. **Tree of Life 10M+**: A comprehensive dataset representing a broad taxonomy of millions of species, with rich hierarchical structure.
2. **iNaturalist (iNat)**: User-generated data collected through citizen science, with species observations from across the world.
3. **Rare Species**: A subset focusing on rare and underrepresented species that are challenging to classify due to data scarcity.

### Dataset Details
- **Intersections and Similarities**: While Tree of Life and iNat have similar species but differ in quality (curated vs. user-generated), the Rare Species subset specifically targets endangered or uncommon species to enhance model robustness.
- **Data Pruning**: Each dataset is cleaned and pruned to remove duplicates, low-quality images, and outliers.

## Model Architecture

### Hierarchical Vision Transformers (Hiera)
**Hiera** is a Vision Transformer (ViT) model specialized in hierarchical classification, which aligns well with taxonomic classification in species identification. By leveraging Hiera, we aim to capture the hierarchical relationships in species data, allowing for more nuanced classification, especially across complex taxonomy levels.

### BioCLIP
**BioCLIP** is a variant of the **CLIP** model, adapted for biological and ecological data. CLIP is a powerful multi-modal model that aligns images and textual descriptions into a shared embedding space. BioCLIP fine-tunes this approach for species classification by leveraging scientific, common, and taxonomic text styles, improving its performance in distinguishing similar or visually ambiguous species.

### Additional Models
Other models, such as standard ViTs, are also tested to provide a comparative baseline for evaluating Hiera and BioCLIP’s performance.

## Methods
The primary methods and preprocessing techniques employed include:
- **Data Augmentation**: Applying RandAug, Mixup, CutMix, and other augmentation techniques to enhance model robustness.
- **Layer-wise Decay**: Utilizing variable learning rates across layers to optimize fine-tuning and model generalization.
- **Text Style Variations for BioCLIP**: Experimenting with different text styles (Common, Scientific, Taxonomic, and Mixed) to determine which performs best for text-image embedding.

## Evaluation Metrics
Two key metrics are introduced to evaluate model performance:
1. **Precision-Recall Based Metric**: Focused on handling the imbalance in species data, especially for rare species.
2. **Embedding-Based Similarity Metric**: Compares the closeness of species embeddings to better capture the model's understanding of species similarity.

## Results
Our results highlight the strengths and weaknesses of each model:
- **Hiera** outperforms standard ViTs in handling hierarchical data structures but requires significant computational resources.
- **BioCLIP** demonstrates strong performance in species classification by leveraging text descriptions, particularly with mixed text styles (scientific, common, and taxonomic).
- **Comparison of BioCLIP vs Hiera**: BioCLIP was more effective when detailed text annotations were available, while Hiera excelled in purely visual classification tasks with a hierarchical structure.

## Lessons Learned
Key insights from the project:
- **Data Imbalance**: Handling rare species remains challenging due to limited data availability.
- **Complex Model Training**: Advanced models like Hiera and BioCLIP require large datasets and computational power.
- **Evaluation Limitations**: Standard metrics were insufficient, necessitating custom metrics tailored to biodiversity tasks.

## Future Work
This project opens up potential research directions:
1. **Refined RAG Methodologies**: Further enhance RAG approaches for improved species classification.
2. **iNat Community Contribution**: Investigate ways to incorporate iNaturalist community contributions for model refinement.
3. **Hybrid Models**: Combine Hiera’s hierarchical capabilities with BioCLIP’s embedding-based approach for more robust results.

## Contributions
Thank you to Graham Taylor for supervising my work at the Vector Institute. Thank you to Nate Lesperance for being a great mentor and advisor. Big Shoutout to the Unviersity of Guelph's Machine Learning Reading Group. Lastly, a huge thanks to the Inaturalist community without whom this project never could've happened.
