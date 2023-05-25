# XGen AI
## Setup
- Clone the project
- Make sure python 3.9 and pip is installed on your system
- Navigate to the main directory containing requirements.txt
- Create a python virtual environment with:
```
python3 -m venv venv
```
- Activate the virtual environment with:
```
source venv/bin/activate
```
- Install the requirements with:
```
pip install -r requirements.txt
```
### Running the script for downloads the images

- Once requirements are installed, run the script using:
```
python download_images.py
```
### Running web app

-  Run the server using
```
flask run
```
- The server is now running on http://127.0.0.1:5000
- The endpoint of interest is http://127.0.0.1:5000/predict

### API Working
#### Request
​
Endpoint: /predict
Method: POST
​
Request:
```
{
    "image": "file", (required)
}
````
​
Response:
```
{
    "data":  {"input":"selected image", "output":"result[]"}
}
```

### Model training

Firstly, download the images
```
python download_images.py
```
Now train the model 
```
cd scripts
python model_train.py
```


### Note
-  Currently my models train on the 1000 images on the given dataset, because due to lack of the resources.



