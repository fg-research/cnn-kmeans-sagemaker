# CNN-KMeans SageMaker Algorithm
The [Time Series Clustering (CNN-KMeans) Algorithm from AWS Marketplace](placeholder) 
performs time series clustering with an unsupervised convolutional neural network trained using contrastive learning 
followed by a K-Means clusterer.
It implements both training and inference from CSV data and supports both CPU and GPU instances.
The training and inference Docker images were built by extending the PyTorch 2.0 Python 3.10 SageMaker containers.
The Docker images include modifications of software licensed under the Apache License 2.0, see the [LICENSE](https://github.com/fg-research/cnn-svc-sagemaker/blob/master/LICENSE) and [NOTICE](https://github.com/fg-research/cnn-svc-sagemaker/blob/master/NOTICE).

## Model Description
### Model Architecture
The model has two components: an encoder which extracts the relevant features, 
and a K-Means clusterer which takes as input the extracted features and predicts the cluster labels.

The encoder includes a stack of exponentially dilated causal convolutional blocks, followed by an adaptive max pooling layer and a linear output layer.
Each block consists of two causal convolutional layers with the same dilation rate, each followed by weight normalization and Leaky ReLU activation.
A residual connection is applied between the input and the output of each block, where the input is transformed by an additional convolutional layer with a kernel size of 1 when its length does not match the one of the output.

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/cnn-encoder-diagram.png style="width:80%;margin-top:30px;margin-bottom:20px"/>

*Encoder architecture (source: [doi.org/10.48550/arXiv.1901.10738](https://doi.org/10.48550/arXiv.1901.10738))*

The encoder parameters are learned in an unsupervised manner by minimizing the triplet loss.
The contrastive learning procedure makes the extracted features of a given sequence (anchor or reference) 
as close as possible to the extracted features of this same sequence's subsequences (positive samples) 
and as distant as possible from the extracted features of other sequences (negative samples).
All (sub)sequences are sampled randomly during each training iteration.

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/cnn-encoder-sampling-diagram.png style="width:50%;margin-top:30px;margin-bottom:20px"/>

*Contrastive learning (source: [doi.org/10.48550/arXiv.1901.10738](https://doi.org/10.48550/arXiv.1901.10738))*

The number of features extracted by the encoder is determined by the number of hidden units of the linear output layer.
These features are used for training the K-Means clusterer. 

**Model Resources:** [[Paper]](https://doi.org/10.48550/arXiv.1901.10738) [[Code]](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries)

## SageMaker Algorithm Description
The algorithm implements the model as described above with no changes.

### Training
The training algorithm has two input data channels: `training` and `validation`. 
The `training` channel is mandatory, while the `validation` channel is optional.

The training and validation datasets should be provided as CSV files.
The CSV file should not contain any index column or column headers. 
Each row of the CSV file represents a time series, while each column represents a time step.
The time series can have different lengths and can contain missing values.
The time series are scaled internally by the algorithm, there is no need to scale the time series beforehand.

See the sample input files `train.csv` and `valid.csv` in the `data/training/` folder.
See `notebook.ipynb` for an example of how to launch a training job.

#### Hyperparameters
The training algorithm takes as input the following hyperparameters:
- `clusters`: `int`. The number of clusters (default = 8).
- `algorithm`: `str`. The clustering algorithm (default = "lloyd").
- `blocks`: `int`. The number of blocks of convolutional layers	(default = 10).
- `filters`: `int`.	The number of filters of all but the last convolutional layers (default = 40).
- `kernel-size`: `int`.	The size of the kernel of all non-residual convolutional layers (default = 3).	
- `reduced-size`: `int`. The number of filters of the last convolutional layer (default = 160).	
- `output-size`: `int`.	The number of hidden units of the linear output layer (default = 320).
- `negative-samples`: `int`. The number of negative samples used for calculating the triplet loss (default = 10).
- `lr`: `float`. The learning rate used for training (default = 0.001).
- `batch-size`: `int`. The batch size used for training (default = 10).
- `epochs`: `int`. The number of training epochs (default = 100).

#### Metrics
The training algorithm logs the following metrics:
- `train_loss`: `float`. Training loss.
- `train_score`: `float`. Training Silhouette Coefficient.

If the `validation` channel is provided, the training algorithm also logs the following additional metrics:
- `valid_loss`: `float`. Validation loss.
- `valid_score`: `float`. Validation Silhouette Coefficient.

See `notebook.ipynb` for an example of how to launch a hyperparameter tuning job.

### Inference
The inference algorithm takes as input a CSV file containing the time series.
The CSV file should not contain any index column or column headers.
Each row of the CSV file represents a time series, while each column represents a time step.
The time series can have different lengths and can contain missing values.
The time series are scaled internally by the algorithm, there is no need to scale the time series beforehand.
See the sample input file `test_data.csv` in the `data/inference/input` folder.

The inference algorithm outputs the predicted class labels, which are returned in CSV format.
See the sample output file `batch_predictions.csv` in the `data/inference/output/batch` folder.

See `notebook.ipynb` for an example of how to launch a batch transform job.

#### Endpoints
The algorithm supports only real-time inference endpoints. The inference image is too large to be uploaded to a serverless inference endpoint.

See `notebook.ipynb` for an example of how to deploy the model to an endpoint, invoke the endpoint and process the response.
See the sample output file `real_time_predictions.csv` in the `data/inference/output/real-time` folder.

## References
- J. Y. Franceschi, A. Dieuleveut and M. Jaggi, "Unsupervised scalable representation learning for multivariate time series," in *Advances in neural information processing systems*, vol. 32, 2019, [doi.org/10.48550/arXiv.1901.10738](https://doi.org/10.48550/arXiv.1901.10738).
