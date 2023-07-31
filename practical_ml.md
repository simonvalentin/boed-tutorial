# Practical Recommendations for Step 4
### Start simple
It makes sense to start with simple models, like a (regularized) linear model. 
In the context of a neural network, this can mean starting with a simple single-layer model, without any hidden layers. 
This serves as a baseline for further experiments, and also to make sure the pipeline is working correctly. 

### Evaluate on unseen simulated data
Always make sure to evaluate on data that was unseen during model training. 
Typical splits, like $60\%$ of data for training, $20\%$ for validation and $20\%$ for testing. 

### We can always simulate more training data
Often, we find that simply simulating more training data provides the biggest difference in performance. 
Especially as we might explore more expressive models (e.g., larger neural networks), we need to keep in mind that these models require more data to train properly.
[See the bitter lesson.](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

### Set up an appropriate experimentation environment
Training a good machine learning model takes some tinkering, and, as any experimentation, requires a way of tracking results.
We thus recommend taking some time to plan out experiments and what they should indicate. 

### Increase complexity incrementally
Increases in complexity, such as more layers, different learning rates or other hyperparameters can and should, in most cases, be done individually. 
The settings we have used for our experiments may provide a useful starting point after exploring a simple linear model. 

### Take care of data normalization issues
Some experiments may generate data whose values vary on a large scale, such as reaction times. 
In these cases, standard pre-processing techniques like $z$-standardization may be useful to allow for better optimizer performance. Make sure to avoid leaking from the validation/test sets to the training set, by computing the standardization statistics only on the training set and applying the transformation to all sets.

### Leverage advancements in automated machine learning
The process of desgining good machine learning models is getting easier and easier with improvements in methods. 
What is likely to stay difficult and domain specific is formulating the "right" scientific questions and building good computational implementations of theories of behavioral phenomena. 

### Make use of known summary statistics
There are often known summary statistics of the raw data that can be used as features for the machine learning models. 
This summary statistics may not necessarily be sufficient, but can serve as a baseline for the summary statistics learned from the raw data. 
For example, we can train a simple linear model (as suggested for the raw data above) on known summary statistics to obtain a baseline level of performance. 
