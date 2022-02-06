#!/bin/bash

# Create virtualenv if necessary
if ! [[ -d "venv" ]]
then
    python -m venv venv/
fi

# Enable the virtualenv
source venv/Scripts/activate

# Install the dependencies if needed
python -m pip install -r requirements.txt

# Update pip package
python -m pip install pip --upgrade