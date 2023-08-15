# food_classification
tested: ![Tested](https://img.shields.io/badge/Arch_Linux-1793D1?style=for-the-badge&logo=arch-linux&logoColor=white)  ![Tested](https://img.shields.io/badge/Debian-A81D33?style=for-the-badge&logo=debian&logoColor=white)  ![Tested](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)  
  
![](https://img.shields.io/badge/python-3.8-blue) ![](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) ![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)   


### Backend Installation  
Manual with docker:  

```shell
docker build <server Name> .
docker run -p 5000:5000 <server Name>
```

Manual with virtualenv:  

```shell
git clone git@github.com:ACT-HealthWatch/food_classification.git  
python3 -m venv <project name>  
rsync -av --exclude food_classification <project name>  
cd <project name>  
pip3 install -r requirements.txt  
gunicorn app:app
```