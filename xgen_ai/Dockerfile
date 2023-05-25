# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3.9

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install python3-pip


RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata curl ca-certificates fontconfig locales \
    && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

#Set an environment variable for the Rails app root folder
ENV APP_ROOT /var/www/PROJECT

#Create the working directory
RUN mkdir -p $APP_ROOT

WORKDIR $APP_ROOT

COPY . ./
RUN pip install -r docker_requirements.txt 

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

EXPOSE 5000
