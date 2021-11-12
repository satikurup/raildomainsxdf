FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD app.py
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6 -y
