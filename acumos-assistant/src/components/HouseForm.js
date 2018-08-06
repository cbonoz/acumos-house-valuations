import React, { Component } from 'react';

import Form from "react-jsonschema-form";

const schema = {
    title: "Enter your Property Details",
    type: "object",
    properties: {
        baths: { type: "number", title: "Baths", default: 1 },
        beds: { type: "number", title: "Beds", default: 1 },
        sf: { type: "number", title: "Square Footage", default: 16000 },
        prop_type: { type: "string", title: "Property Type", default: "Other" },
        year_built: { type: "number", title: "Year Built", default: 2000 },
        lot_size: { type: "number", title: "Lot Size", default: 100 },
        hoa: { type: "number", title: "Home Owner Association fees", default: 10000 },
        dom: { type: "number", title: "Days on Market", default: 10 },
        location: { type: "string", title: "Location", default: "Cambridge" },
        state: { type: "string", title: "State", default: "MA" },
        city: { type: "string", title: "City", default: "Boston" }
    }
};


class HouseForm extends Component {


    componentWillMount() {
        this.state = {
            value: "",
            loading: false
        }
    }

    getValue(data) {
        const self = this;
        const url = "http://localhost:3001/predict"
        const payload = data.formData;
        self.setState({loading: true});
        fetch(url, {
            method: 'POST', // or 'PUT'
            body: JSON.stringify(payload), // data can be `string` or {object}!
            headers:{
              'Content-Type': 'application/json'
            }
          }).then(res => res.json())
          .catch(error => {
              self.setState({value: "", loading: false})
            console.error('Error:', error)
          })
          .then(response => {
              console.log('Success:', response)
              self.setState({loading: false, value: response['prediction']});
          });
    }

    render() {
        const self = this;
        const value = self.state.value;
        const loading = self.state.loading;
        return (
            <div>
                <div className="form-area">
                    <h2 className="centered green header-text">Your Instant Property Appraisal</h2>
                    {(value == "" && !loading) && <Form schema={schema}
                        onChange={console.log("changed")}
                        onSubmit={(data) => {
                            self.getValue(data)
                        }}
                        onError={console.log("errors")} />}
                        {loading && <h3>Loading...</h3>}
                    {(value != "" && !loading) && <div><p className="centered">Estimate: <span className="green">{value}</span></p></div>}
                </div>
            </div>
        );
    }
}

export default HouseForm;