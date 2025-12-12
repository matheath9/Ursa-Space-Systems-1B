![image alt](https://github.com/matheath9/Ursa-Space-Systems-1B/blob/main/img/home-card-min-1024x582.jpg?raw=true)

# 🚢🧊 Foundation Models for Satellite Image Intelligence
  
**URSA Space Systems – Collaborative Research Project** 

- SAR Image Classification: Icebergs vs. Vessels 


This is where code will be stored throughout the Ursa Space 1B project. 

---

### 👥 **Team Members**


| Name               | GitHub Handle     | Contribution                                          |
| ------------------ | ----------------- | ----------------------------------------------------- |
| **Shahzadi Aiman** | @ShahzadiAiman    | Model development, model research, feature extraction |
| **Matan Heath**    | @matheath9        | Model development, embedding visualization            |
| **Stella Li**      | @stellali28       | Model research, feature extraction                    |
| **Asheni Agarwal** | @asheniagr        | Exploratory data analysis (EDA)                       |
| **Hinna Zeejah**   | @hinnazeejah      | Feature extraction                                    |
| **Colin Emmanuel** | @Colin-J-Emmanuel | Model development, embedding visualization prep       |
| **Favour Umejesi** | @favour-umejesi   | Feature extraction                                    |


---

## 🎯 **Project Highlights**

- Developed a machine learning model using baseline algorithms (MLP, Random Forest, Logistic Regression, CNN, Gradient Boosting) and embedding-based models using ViT and ConvNeXT to address iceberg vs. vessel classification in SAR imagery.
- Achieved performance improvements when using pretrained-model embeddings compared to baseline models, demonstrating the value of transfer learning for enhanced maritime object identification for URSA Space Systems.
- Generated actionable insights through exploratory data analysis to inform modeling decisions and guide preprocessing choices
- Implemented a baseline-vs-pretrained comparison methodology to meet industry expectations for reliability, interpretability, and performance benchmarking in satellite image analysis.

---

## 👩🏽‍💻 **Setup and Installation**

**How to clone the repository**


Run the following commands in terminal:

	git clone <https://github.com/matheath9/Ursa-Space-Systems-1B.git>
	cd Ursa-Space-Systems-1B

<br>

**How to install dependencies**


This project uses Python. This project uses Python 3 so make sure it is installed .  
Then install required packages:

	pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras torch torchvision jupyter pillow kaggle
	
Note: Sometimes one package may fail due to version conflicts or missing system dependencies. If that happens, you can install the problematic package separately.

<br>

**How to access the dataset(s)**


The dataset used in this project comes from the Kaggle competition Statoil Iceberg Classifier Challenge.

- Sign in to Kaggle (create an account if you don’t have one).
- Go to the competition dataset page: https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data
- Download the dataset files.
- Place the downloaded files into the data/ folder in the project directory.

Optional: If you have the Kaggle API set up, you can also download the data programmatically:

Run the following commands in terminal:

	kaggle competitions download -c statoil-iceberg-classifier-challenge
	unzip statoil-iceberg-classifier-challenge.zip -d data/


---

## 🏗️ **Project Overview**

This project is part of the **Break Through Tech AI Program**, completed during the AI Studio portion where fellows collaborate with an industry partner to solve a real-world, AI-driven problem. As part of our team workflow, we focused on understanding the dataset, developing machine learning models, and evaluating their performance to identify the most effective approach.

Our project objective centered on building and comparing multiple ML models to understand how different architectures perform on the assigned dataset. The scope included exploratory data analysis (EDA), data preprocessing, model building, embedding extraction, and performance comparison across models contributed by each team member.

The real-world significance of this work stems from the importance of selecting efficient, well-generalized models when working with image data. By comparing baselines to more advanced approaches using pre-trained embeddings, we gained insight into how model selection affects accuracy, training time, and overall performance—knowledge that translates directly to practical applications in computer vision and scalable AI systems.

---

## 📊 **Data Exploration**

We began with exploratory data analysis (EDA) to understand the dataset’s structure and class distribution. This included inspecting sample images, checking pixel brightness and variance, and examining overall patterns within the dataset.

**Dataset Overview**

- Origin: Kaggle competition Statoil Iceberg Classifier Challenge
- Format: CSV file containing JSON-encoded radar images; separate training and test sets
- Size:
	- Training set: ~1604 labeled training samples
	- Test set ~8424 unlabeled test samples
- Type of data: Synthetic Aperture Radar (SAR) grayscale images of icebergs and ships, with labels for classification
- Label
	- 0 = Ship
 	- 1 = Iceberg
- Image Format:
	- 2-channel SAR images (band_1, band_2)
 	- Image size_ 75 x 75 pixels
- Pixel Range (After Normalization)
	- Values scaled to [-1, +1]
  	- -1 -> darker return
  	- +1 -> brighter return


	
**Insights from EDA**

- Classes (icebergs vs ships) were fairly balanced
- Some images were noisy or low-resolution, highlighting the challenge of distinguishing icebergs from vessels

	
**Sample dataset images:** 

![Sample Images](img/sample-images-of-dataset.png) 



**Dataset Sample:** 

A preview of the SAR dataset containing radar image bands and the corresponding iceberg labels.

![Dataset Sample (SAR Bands and Labels)](img/Dataset%20Sample%20(SAR%20Bands%20and%20Labels).png)



---

## 🧠 **Model Development**
  
**Models used:** 

- Built baseline MLP model and explored various other models such as CNN, logistic regression, Random Forest, etc.
- Generated image embeddings using pre-trained models from HuggingFace:
	- ViT model: https://huggingface.co/google/vit-base-patch16-224
	- ConvNeXT model: https://huggingface.co/docs/transformers/en/model_doc/convnext
 
Note: The HuggingFace models were not specifically trained on SAR datasets

<br>

**Architecture of Baseline MLP Model**

- Input: 11,250 features (75×75×2 flattened)
- Hidden Layers: 512 → 256 → 128 neurons
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Framework: scikit-learn MLPClassifier

<br>

**Training Configuration of Baseline MLP Model**

- Training samples: 1,283 (80%)
- Validation samples: 321 (20%)
- Batch size: 32
- Early stopping: 15 epochs patience
- Converged in: 53 iterations
- Final loss: 0.0034

---

## 🧩 Code Highlights ##

**Main/**

Final consolidated notebooks and project-wide workflows used for results, figures, and summary analysis.

**Experimental Notebooks/**

Individual team member exploration:

- Asheni: Random Forest experiments
- Matan: CNN and Gradient Boosting models
- Shahzadi: Logistic Regression + Experimental MLP model
- Stella: Data loading/testing notebook

**Project/**

Core project files used for the final pipeline:

- Code/ : Master notebook running all models + embedding split utilities
- Data/ : Saved embedding files for ViT and ConvNeXt
- Img/ : Final figures (PCA plots, confusion matrix, dataset samples)
- README.md :Project-level documentation

**ConvNext/**

All work related to ConvNeXt embeddings:

- Notebook for loading data
- Embedding generation script
- Test scripts and saved embeddings

**Baseline-models/**

Contains the baseline modeling work.
This folder is intended to store the final selected baseline MLP model.
At present, it includes earlier baseline experiments (Logistic Regression and an initial MLP), while the final baseline MLP notebook was pending teammate upload at the time of submission.

**embeddings (ViT/)**

Saved ViT embeddings + notebook for loading/testing them.

**Model-results/**

Stored confusion matrices, screenshots, and performance output files for final evaluation.


---

## 📈 **Results & Key Findings**

**Baseline MLP model:**

Performance Metrics:
- ROC-AUC Score: 0.9855 (98.5%)
- Accuracy: 0.96 (96%)

Key Results:
- 12 errors out of 321 (3.7%)
- 306 high-confidence predictions
- Balanced performance across classes

Confusion Matrix: 
![Confusion Matrix](img/Baseline-model-confusion-matrix.png)

<br>

**Embeddings**

Vizualizations of seperation and variance using 3D PCA:

![3D PCA Embeddings](img/3D-PCA-embeddings.png)

Seperation Ratios:
- Vit seperation ratio: 0.475 
- Convnext seperation ratio: 0.537

<br>

**Final Thoughts:**
Although the embedded data still modeled well, it was not enough compared to the baseline.
For iceberg vs. ship classification:  baseline model meets the business need



---

## 🚀 **Next Steps**

**What to improve with more time/resources**
- Try deeper or more specialized CNN architectures.
- Add cross-validation for more reliable evaluation.

<br>

**Future directions**

- Incorporate more SAR datasets.
- Explore pretrained models designed for satellite imagery.
- Consider adding metadata or multi-modal inputs.


---

## 📝 **License**

This project is licensed under the MIT License.

---

## 🙏 **Acknowledgements** 

Thank you to our Challenge Advisor, Nick LaVigne, for his guidance and feedback throughout this project.  
A special thanks as well to our coach, Eric Bayless, for his support and direction during the challenge.

