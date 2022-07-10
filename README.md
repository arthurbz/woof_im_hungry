# Woof 🐶 I'm hungry

Super simple AI model to check if my dog has food in his bowl.

<h3> Installing all requirements </h3>

Create a virtual enviroment: <br>
```python -m venv venv``` <br>

Activate it: <br>
```source venv/bin/activate``` <br>

Install requirements: <br>
```pip install -r requirements.txt```

<h3> Generating Dataset </h3>

- Video with an empty bowl: ```videos/empty/```
- Video with a full bowl: ```videos/full/```

Then we can run: <br>
```python generate_dataset.py```

<h3> Train Model </h3>

Now we have three subdirectories on ```dataset``` directory: train, validation and submission. Each of them has a subset of the whole dataset. <br>
With the dataset ready, we can train the model with: <br>
```python model_train.py```

<h3> Checking how good it is </h3>
After we trained our model we can check its results through a script, which will show each of the submissions
(that can be found on dataset/submissions/) and the prediction of our model. This will also create submissions.csv with the results. <br>
We can do that by running:

```python model_train.py``` <br>

<h3> Done! </h3>
