# Show and tell
This project implements <a href = "https://arxiv.org/abs/1411.4555"> the Show and Tell paper </a>
In simple words, it takes an image and produce an English sentence that describes the image.  

# Model Short Summary
Resnet: takes an image and produce a 300-dimension image embedding vector
LSTM: takes the image embedding vector above and a start word token(STK) to produce a prediction for next word (as a probability distribution)
Beam Search: Find the optimal sequence of words given by the probability distribution given at each time step(optimal only under some beam size, not optimal in general).


# Some Screenshots
![Alt text](/screenshots/ss1.png?raw=true "ss1")

