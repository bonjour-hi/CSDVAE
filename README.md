# CSDVAE
Using Collaborative Stacked Denoising Variational Autoencoders to implement recommendation system tasks.
* GPU: GeForce RTX 2080Ti ; Time: test epoch 50 sec; train epoch 7-10min<br>

### The function is introduced as follows:<br>
- Run main.py to run the project.<br>
- data_preprocessor.py is used to solve the data reading problem.
- data_wash.py is used to convert the .csv dataset to the prescribed .txt format, and complete the work of dividing the training set and the test set.
- DAE.py is used to establish variables, or choose to pretrain.
- CSDAVE.py is responsible for the main model CSDVAE.
