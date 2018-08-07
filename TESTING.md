Acumos Property Assistant
---

A python-powered machine learning model for determining the valuation of your property - powered by Acumos, Redfin, and sklearn.

## Testing

Two ways of testing:

### Running the model locally on a static csv input.

The python notebook (ipynb) has been converted to a executable python file and will generate the test predictions for the full csv file.

<pre>
    pip3 install -r requirements.txt # or via pipenv
    python3 acumos_property_assistant.py
</pre>

The results should now be in the file `assets/active_predictions.csv`.

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
