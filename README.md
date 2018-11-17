# Show and tell
This repo is to reimplement the show and tell paper.

# To Do
* BLEU_eval()
* sample_images_and_save_on_tensorboard()
* Tune hyperparameters.
	* Data augmentation?
	* Optimizer, learning rate, weight decay?
	* Clip gradients?
		* [Github link](https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L161-L164)
		* [Pytorch forums](https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/14)
	* Which ResNet?
	* Which RNN: LSTM? GRU?
	* Word embeddings:
		* Embedding size?
		* Vocabulary size?
	* RNN parameters?
		* Hidden size?
		* Number of layers?
		* Dropout?
	* Beam search size?
	* Google open source parameter settings of the NIC, in tensorflow:
		* [Github link](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/configuration.py)
* Train one week!

