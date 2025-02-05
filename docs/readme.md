# Flight Data Analysis and Classification

This project implements an end-to-end data pipeline for analyzing and classifying flight characteristics using ADS-B Exchange historical data. The analysis combines web scraping, data processing, and machine learning techniques to uncover patterns and insights in flight behavior.

## Project Overview

Hello, I'm so excited for you to be looking at my project! Let me show you around.

If you want to get a high level overview, you're in the right place, just keep reading.

If you want to get a deeper understanding of the project then what is here, then you can jump straight to the notebooks by following these links: 
- https://x81k25.github.io/ADSB_classification/blob/main/docs/01_data_engineering_EDA.html
- https://x81k25.github.io/ADSB_classification/blob/main/docs/02_feature_engineering.html
- https://x81k25.github.io/ADSB_classification/blob/main/docs/03_ML_pipeline.html
- https://x81k25.github.io/ADSB_classification/blob/main/docs/04_analysis_conclusions.html
 
Oh, that's not enough, you want to run the notebooks yourself? If you are accessing the file from github, the full repo can be found here:

- https://github.com/x81k25/ADSB_classification

Just clone the repo and get going. Everything for the notebooks is in here.

If you still want to get deeper and fully recreate this project, then keep reading. The requirements to run the full project are detailed below.

## Repository Structure

```
.
├── data/              # Data files included with the repository
├── notebooks/         # Jupyter notebooks for analysis and visualization
├── scripts/           # Python scripts for data pipeline
├── sql/               # SQL scripts for database setup
├── requirements.txt   # Python dependencies
└── template.env       # Environment variable template
```

## Setup Instructions

1. Clone the repository
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required Python packages:

```bash 
pip install -r requirements.txt
```

Notes on requirements:
- currently requirements.txt contains all packages to run all portions of the project
- the requirements.txt is broken down into dependencies for the scripts and notebooks separately but some of the scripts requirements are also required for the notebooks
- if you have no plan to run PyTorch with GPU acceleration, you should be able to delete the pytorch section of the requirements.txt and do a standard pip install of the torch package
  - i have not tested this method, but all code is written to fall back to CPU if the GPU requirements are not detected

4. Register venv as ipykernel:

```bash 
python -m ipykernel install --user --name=venv --display-name=ADSB_classification
```

## Conclusions and Results

TBA

## Methodology

### Problem Statement

The overall problem I am trying to solve within this project is simply to answer the question:  create classifications of flights for future analysis, with completely unlabeled ADSB flight data. 

### Data Selection

All data is derived from: https://www.adsbexchange.com/products/historical-data/

For this proces I mainly use the trace files, although I do some analysis on the readsb-shit data s well. I originally planned on used readsb-hist data for the whole project, as on first inspection it seemed adequately large and of sufficiently high quality to answer the fundamental questions within the problem statement. 

However, after importing and performing EDA on the readsb-hist data, I realized it was not well suited for my use case. The reason is because, my ML pipeline is going to work well with a large sample size of detailed trace data from individual flights. When I recreated flight trace data using readsb-hist, the traces assembled had relatively few consecutive data points and short duration.

All of the problems are solved by the using the traces data. The traces data contains tens of thousands of complete and detailed traces that are over 15 minutes and have many data points along the flight path. The hires-traces data may have been even more interesting, but the traces dataset is more than sufficient for this project.

Through the use of parameters the scripts contained in this folder can use any of the historical data of the appropriate type from the data source above, the default params in this project all use data from August 1st, 2024 The ML methodology I am using works best with a large sample size of data that on aggregate is as normal as possible. Here are the details of my thought process:
- relatively recent data
- it's a Thursday (mid-week)
- it's not near any major global holidays
- August typically has stable weather patterns globally
- most of Europe is on vacation, but their flights are replaced by leisure travelers, keeping volume similar
- it's outside of major conference/convention seasons
- schools aren't typically starting yet in most regions
- business travel is relatively normal in most regions

It is important to note that the data given is incomplete, and I did not find documentation on exactly how it is subsetted. For example, it is obvious that data for only the first day of the month is given for all fo the data sets. However, exactly I did not find explicit information about which flights are shown in the sample data provided. Because of this ambiguity, many data quality checks are performed throughout the pipeline. Within the 24 hour window used, sufficient traces are given, that traces with data quality issues can be omitted while keeping the overall sample size sufficient.

More information about the Methodology, including the feature selection and feature generation, will be contained in the jupyer noteboks in the notebooks folder.

### Model Selection

#### Supervised vs Unsuperivsed Learning

Key Advantages of Supervised Learning:

- Results are directly interpretable since each class has a known real-world meaning
- Can leverage existing domain expertise through the labeling process
- Performance can be quantitatively validated against ground truth

Key Advantages of Unsupervised Learning:

- Can discover novel patterns that weren't previously known or considered
- Can analyze the complete dataset without being limited by label availability
- Not limited by the potential quality issues or obtuseness of labels

An unsupervised learning approach lets the data tell its own story. Rather than forcing the data into predefined boxes, we're letting the model make its own findings, discovering the true nature of contend of the data in ADSB records.

#### Potential Unsupervised Approaches

Deep Learning Models

- Autoencoder + HDBSCAN
  - Pros:
    - Simultaneously learns normal flight patterns while providing rich anomaly detection through reconstruction error
    - Produces interpretable latent space representations that help explain both clusters and anomalies
    - Can detect both global pattern deviations and local anomalies (like sudden accelerations) in a single model
  - Cons:
    - Requires significant tuning of both autoencoder architecture and HDBSCAN parameters
    - May overfit to common flight patterns if training data isn't sufficiently diverse
    - Computationally intensive during training phase compared to traditional methods

- GANs
  - Pros:
    - Can learn very subtle deviations from normal flight patterns
    - Particularly good at handling multimodal distributions in flight behaviors
    - Can generate synthetic examples to help validate anomaly detection
  - Cons:
    - Notoriously difficult to train and achieve stable convergence
  - Much higher computational overhead than other approaches
    - May suffer from mode collapse, missing important but rare flight patterns

- LSTM-based approaches
  - Pros:
    - Naturally handles temporal dependencies in flight patterns
    - Excellent at predicting expected behavior sequences
    - Can capture long-term dependencies in flight phases
  - Cons:
    - Tends to focus on global patterns, potentially missing local anomalies
    - Memory requirements scale with sequence length
    - Can be biased toward more common flight patterns, potentially normalizing subtle anomalies

Probabilistic Models 

- Hidden Markov Models (HMMs)
  - Pros:
    - Naturally models flight phase transitions
    - Computationally efficient once trained
    - Very interpretable state transitions
  - Cons:
    - Struggles with continuous, high-dimensional flight data
    - Limited ability to detect subtle anomalies
    - Can be biased by initial state assumptions

Traditional Machine Learning Models

- Traditional Clustering (K-means/DBSCAN)
  - Pros:
    - Computationally efficient and easy to implement
    - Very interpretable results with clear cluster centroids
    - Minimal hyperparameter tuning needed compared to deep learning approaches
  - Cons:
    - Cannot naturally handle temporal aspects of flight data
    - Struggles with variable-density clusters of flight patterns
    - No inherent mechanism for anomaly detection
- Isolation Forest
  - Pros:
  - Specifically designed for anomaly detection
    - Computationally efficient even with large datasets
    - Works well with high-dimensional flight data
  - Cons:
    - No clustering capability for normal flight patterns
    - Cannot capture temporal relationships in data
    - Limited interpretability of why something is classified as anomalous

And the winner is: ...we're going with the Autoencoder + HDBSCAN combination because it uniquely provides both rich anomaly detection capabilities through reconstruction error and effective clustering of normal flight patterns in a single model architecture, while maintaining interpretability through its latent space representations. This approach outperforms other models by simultaneously capturing both global flight patterns and local anomalies (such as unusual accelerations or trajectory deviations), and though it requires more computational resources during training, the quality of insights it provides justifies this tradeoff compared to simpler approaches.

## Components

### data

Here I store the data required for the execution of the jupyter notebooks. The entirety of the data set used was big, for the largest table over 100 million rows was ingested. Most viewers of this repository are not going to want to wait around to process all that, so I have placed relevant aggregations, subsets, and objects in the data folder to speed everything up.

Okay, if you really want all of the data, you recreate it entirely yourself. First, clear your schedule for a day or two, then go the Scripts section of the readme and follow the instructions.

### notebooks

- detailed commentary on process and methods chosen
- summaries of all of the processes completed in the scripts
- model analysis
- conclusions drawn from analysis
- some cool visualizations

### scripts

Here is where the vast majority of the code for this projects lives. If you really want to recreate all the steps I did with the full data set, everything you need is in here. Several steps will need to be conducted before the scripts can be initiated.

#### Setup

- you will need a PostgreSQL database with around 15Gb of storage available
  - I am using a PostgreSQL databse running on a local server in a docker container, which is to say you will not need the beefiest PosgreSQL instance, but you're going to need a few cores and at least 4Gb of memory allocated
  - you will also need to have permissions to create tables and insert data in this databsae
  - all of the scripts needed to create the tables are in the sql folder
    - run every sql script in the sql folder
    - they can be executed in any order, as no foreign keys were implemented
  - all credentials are read in via dotenv
    - to enter your credentials, option the template.env, insert your credentials, and then rename the file to ".env"
  - you can ignore the other scripts in the sql folder that do not being with "instantiate"
    - in case you're curious, those SQL scripts are used by various pyhton scripts and are kept there to improve the readability of the python scripts
- you will also need to properly add your project path in the .env file

#### Requirements

- GPU acceleration is utilized, currently only for the autoencoder training
  - if you do not have a GPU, the scripts will default to CPU without any code alterations
  - the requriements.txt file is configured for Cuda 11.8; if you wish to utilize GPU acceleration you will need to check your hardware and update requirements.txt appropriately 
- Memory
  - 32Gb available system memory
- CPU
  - all threaded activities are dynamically set based on core count
  - I am using 16, so any estimations of time given below will be based on that
- time
  - acceleration, parellelization, thread pooling, and various other optimizations were used for the creating of the scripts in this repository
  - ...but, as this is fundamentally an academic projects and will not ultimately be used in a production environment, some of the scripts still take a long time
  - depending on your hardware, you may have to run some of the scripts overnight
  - the most time consuming scripts will be _04_traces_to_db.py and _06_traces_db_partitioning.py
  - most scripts will execute in <1 hour, and some will only take minutes

#### Script execution

- all of the scripts are configured to run via CLI with default arguments,e.g.:
```bash
cd <project_dir> 
venv/bin/activate
cd scripts
python _01_readsb_hist_to_db.py
```
- the scripts are designed to be run in sequence, as indicated by the firs 3 characters, i.e.: _xx
- before you run the script straight out!
  - each script is equipped with a test mode, .i.e.:
```bash
python _01_readsb_hist_to_db.py --test
```
  - the test mode will run a with minimize parameters, the intent is to:
    - confirm that the script is working functionally in your environment
    - give you an idea of how long the script will take; for the looger exeuction times scripts, most of the test mode configuration represent about 1/256 of the whole data set
- most of the important script parameters are contained within the main function and can be either altered by CLI, or if you wish altered by changing the default args within the scripts

#### Script contents

Note on script contents: Scripts _01 -> _03 are not necessary to run the following scripts. Scripts _04 -> _10 must be executed sequentially. Scripts 1-3 used the readsb-hist data set, which only after EDA I discovered where actually much less rich dataset than what was contained in the trace data. The data contained within could still be useful if you wish to perform further analysis on this subject area, but it is not required for the ML pipeline used in the later scripts. 

- _01_readsb_hist_to_db.py
 - Fetches aircraft tracking data from ADS-B Exchange in 5-minute intervals using parallel processing
 - Transforms raw aircraft data (position, altitude, speed, etc.) into a structured format
 - Loads processed data into database using batch inserts with duplicate handling
 
- _02_readsb_hist_to_db.py
 - Analyzes aircraft tracking data using DuckDB as an intermediate connection layer
 - Performs table analysis by calculating basic statistics including row counts, column data types, null values, and distinct values
 - Analyzes correlations between numeric columns with configurable methods (Pearson, Spearman, or Kendall)
 - Parameters include sample size, correlation method, and correlation threshold

- _03_readsb_hist_trace.py
 - Transforms raw tracking points into coherent aircraft traces by grouping data points from the same aircraft within a configurable time gap (default 600 seconds)
 - Uses chunked processing approach (default 1 hour chunks) to handle large datasets
 - Creates structured trace objects containing flight information (aircraft ID, flight number, registration) and trajectory data
 - Processed traces are stored with conflict handling to prevent duplicates

- _04_traces_to_db.py
 - Processes aircraft trace data from ADS-B Exchange, downloading JSON files organized by hex values and dates
 - Uses asynchronous operations and parallel processing with configurable batch and chunk sizes
 - Pipeline includes validating hex ranges, fetching directory listings, downloading files, and deduplicating records
 - Schema captures various aircraft parameters like position, speed, altitude, and technical details

- _05_traces_EDA.py
 - Uses DuckDB to analyze aircraft trace data
 - Statistical analysis includes total row counts, top aircraft by trace points, mean point counts per aircraft, and duration-based statistics
 - Configurable parameters for point count limits and minimum duration filters

- _06_traces_db_partitioning.py
 - Partitions aircraft tracking data table into smaller tables based on first two characters of ICAO aircraft identifier (hex prefixes '00' to 'ff')
 - Creates up to 256 separate tables with identical schema
 - Transfers data in batches of 100,000 rows with indexes on timestamp and ICAO columns
 - Can process either full hex range (00-ff) or specific subsets

- _07_traces_to_segments.py
 - Processes aircraft tracking data across different hex-partitioned tables
 - Implements multi-level parallelization: thread pooling for hex partitions and process pooling for aircraft (ICAO codes)
 - Three main stages: extraction of raw trace data, transformation of flight segments based on timestamp gaps, and loading of processed data

- _08_trace_feature_generation.py
 - Processes flight parameters from aircraft_trace_segments table
 - Extracts distinct ICAO codes, retrieves segments for each aircraft
 - Transforms raw data into statistical features (mean, std, min, max, median)
 - Handles multiple data types (decimals, integers, strings, timestamps) for key flight parameters

- _09_autoencoder_training.py
 - Implements autoencoder neural network using PyTorch to compress high-dimensional flight metrics
 - Model architecture: encoder and decoder with three layers each
 - Uses StandardScaler to normalize features including flight metrics, duration, hour of day, and day of week
 - Training uses Adam optimizer and Mean Squared Error loss

- _10_clustering.py 
 - Performs flight pattern clustering using HDBSCAN algorithm
 - Workflow: data loading from pickle file, preprocessing (outlier removal using z-scores and scaling), and clustering
 - Configurable parameters include minimum cluster size and epsilon
 - Analyzes cluster characteristics including average altitude, speed, vertical rate, and duration"""

### sql

The SQL scripts contained are required only to run the scripts in this repo; the notebooks only need what is already in the data folder; so if your only doing the notebooks you can skip this. As per the instructions in the scripts sections above, run all fo the scripts in this folder to create the necessary tables. They can be run in any order.