# Use the official Python image as a base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR C:/Users/tejav/IdeaProjects/Cloud_proj

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Set the command to run your Flask application
CMD exec C:\\Users\\tejav\\IdeaProjects\\Cloud_proj\\venv\\Lib\site-packages\\gunicorn --bind :$PORT --workers 1 --timeout 120 main:app
