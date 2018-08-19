Acumos Property Assistant
---

A python-powered machine learning model for determining the valuation of your property - powered by Acumos, Redfin, and sklearn.

## Testing

Two ways of testing:

### Running the model locally on a static csv input.

The python notebook (ipynb) has been converted to a executable python file and will generate the test predictions for the full csv file. 

This program will train off the housing data in `assets/redfin_2018_8_boston.csv`, and evaluate the test data (currently listed properties) contained in `assets/test_data.csv`.

<b>Before starting, it's recommended that you are running python 3.6. This was tested using a python3.6 environment.</b>

<pre>
    pip install -r requirements.txt # or via pipenv
    python acumos_property_assistant.py
</pre>

The predictions for the test data should now be in the file `assets/active_predictions.csv`.

### Running the model as a application/server.

Start the web server from the root folder.

<pre>
    pip3 install -r requirements.txt # or via pipenv
    python3 server.py
</pre>

Start the application.

<pre>
    cd acumos-assistant
    yarn && yarn start
</pre>

