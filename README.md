# DisasterTweetAnalytics
## Exploring Changes in Semantic Attributes of Tweets in Response to Natural Disaster Events

**by Aiyana Signer, Diego Gomes, Nathalie Nitsingam & Stylianos Psychias**

This project focuses on how social media responses, specifically those found on Twitter, change in the face of natural disasters. It investigates how semantic attributes of tweets, specifically aggressiveness, sentiment, and stance toward climate change, change when a natural disaster occurs. The process involves the generation of training data from tweets and disaster events, the training of deep learning models, and an analysis of the results.

### Components

#### ``geoprocessing.py``
This Python module provides the core geospatial processing functionality used throughout this project. It defines two main classes: Disaster and TweetRaster. The Disaster class represents a disaster instance, while TweetRaster class helps in rasterizing the Tweet data into a set of pixels, allowing us to handle the data in a spatially aggregated manner.

#### ``geoprocessing_demo.ipynb``
This Jupyter notebook demonstrates the functionality of the geoprocessing module, which includes disaster data loading, tweet rasterization, and data preprocessing. It's a great way to understand the core geospatial processing methods.

#### ``dataset_compiler.ipynb``
This Jupyter notebook is responsible for generating the training data. It loads tweets and disaster data, rasterizes the tweets into a set of pixels, calculates the semantic changes in response to natural disasters, and exports a variety of training datasets for machine learning models.

#### ``model_trainer.ipynb``
This Jupyter notebook is where the deep learning models are defined and trained. Using a Fully Connected Neural Network (or Dense Network), it reads the training datasets created by the dataset_compiler.ipynb notebook, trains models on this data, and evaluates their performance. For each semantic attribute, models are created and evaluated using different time windows around a natural disaster event to understand how these time frames affect predictability.

#### ``exploratory_analysis.ipynb``
This is a placeholder for an upcoming Jupyter notebook that will contain the exploratory analysis of the project.

<br>
<br>

**Please refer to individual files for more detailed explanations and code demonstrations.**