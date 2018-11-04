# Show and tell
This repo is to reimplement the show and tell paper.

# To Do
* Finish two functions/classes in utils.py.
	* beam_search()
		* [Github link](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/inference_utils/caption_generator.py)
	* metrics()
		* [Github link](https://github.com/tylin/coco-caption/tree/master/pycocoevalcap)
* Linux version.
* Debug.
* Tune hyperparameters.
	* Data augmentation?
	* Optimizer, learning rate, weight decay?
		* ADAM overflows in Blue Waters?
	* Clip gradients?
		* [Github link](https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L161-L164)
		* [Pytorch forums](https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/14)
	* Which ResNet?
	* Which RNN: LSTM? GRU?
	* Freeze word embeddings?
		* Embedding size?
		* Vocabulary size?
	* RNN parameters?
		* Hidden size?
		* Number of layers?
		* Dropout?
	* Beam search size?
	* Google open source parameter settings of the NIC, in tensorflow:
		* [Github link](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/configuration.py)
* Train two weeks!

