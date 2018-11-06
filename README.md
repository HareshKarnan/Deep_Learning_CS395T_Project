# Yearbook/ Geolocation
This is a starter code for implementing yearbook or geolocation project.

## Dependencies

We used Docker for all our experiments and reccomend the same. 

The Docker Image we used was `ufoym/deepo:all`

The command to run this docker container is:

`
sudo docker run -it --network host --runtime=nvidia --shm-size 12G -v ~/:/home/ ufoym/deepo /bin/bash
`

This mounts your `/home` into the container's `/home`. Once this is done, cd into the project1 directory (placed wherever it is in your system, accordingly), and run 

`python src/grade.py --DATASET_TYPE yearbook --type test`

# Project Folder Structure
```
data
	yearbook
		train
			F
				000001.png
				...
			M
				000001.png
				...
		valid
			...
		test
			...
		yearbook_train.txt
		yearbook_valid.txt
		yearbook_test.txt
	geo
		train
			000001.JPG
			...
		valid
			...
		test
			...
		geo_train.txt
		geo_valid.txt
		geo_test.txt
model
	TODO: put your final model file in the folder
src
	TODO: modify load and predict function in run.py
	grade.py
	run.py
	util.py
output
	TODO: output the yearbook/geo test file
	geo_test_label.txt
	yearbook_test_label.txt
```

## Evaluation
### Data setup
Download the data from the link and store it in data folder as described in the folder structure.

### Models
Train the model and put the model in the `Model` folder

### Running the evaluation
It will give the result based on the baseline 1 which is the median of the training image.
1. For yearbook validation data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=valid
```

2. For Geolocation validation data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=valid
```

### Generating Test Label for project submission
1. For yearbook testing data
```
cd src &&  python grade.py --DATASET_TYPE=yearbook --type=test
```

2. For Geolocation testing data
```
cd src &&  python grade.py --DATASET_TYPE=geolocation --type=test
```

## Submission
1. Put model and generated test_label files in their respective folder.
2. Remove complete data from the data folder.
3. Add readme.md file in your submission.
4. Project should be run from the grade.py file as shown in the evaluation step and should be able to generate the test_label file.
