import React, { Component } from 'react';

import Form from "react-jsonschema-form";

const schema = {
    title: "Enter your Property Details",
    type: "object",
    properties: {
        cpsf: { type: "number", title: "Cost per square foot", default: 1000 },
        baths: { type: "number", title: "Baths", default: 1 },
        beds: { type: "number", title: "Beds", default: 2 },
        sf: { type: "number", title: "Square Footage", default: 2000 },
        prop_type: { type: "string", title: "Property Type", default: "Other" },
        year_built: { type: "number", title: "Year Built", default: 2000 },
        lot_size: { type: "number", title: "Lot Size", default: 1000 },
        hoa: { type: "number", title: "Home Owner Association fees", default: 1000 },
        dom: { type: "number", title: "Days on Market", default: 10 },
        location: { type: "string", title: "Location", default: "Cambridge" },
        state: { type: "string", title: "State", default: "MA" },
        city: { type: "string", title: "City", default: "Boston" }
    }
};


class HouseForm extends Component {


    componentWillMount() {
        this.state = {
            value: ""
        }
    }

    getValue(data) {
        const self = this;
        const url = "http://localhost:3001/predict"
        const payload = data.schema.properties;
        fetch(url, {
            method: 'POST', // or 'PUT'
            body: JSON.stringify(payload), // data can be `string` or {object}!
            headers:{
              'Content-Type': 'application/json'
            }
          }).then(res => res.json())
          .catch(error => console.error('Error:', error))
          .then(response => {
              console.log('Success:', response)
              self.setState({value: response['prediction']});
          });
    }

    render() {
        const self = this;
        const value = self.state.value;
        return (
            <div>
                <div className="form-area">
                    <h2 className="centered green header-text">Your Instant Property Appraisal</h2>
                    {(value == "") && <Form schema={schema}
                        onChange={console.log("changed")}
                        onSubmit={(data) => {
                            self.getValue(data)
                        }}
                        onError={console.log("errors")} />}

                    {(value != "") && <div><p className="centered">Estimate: <span className="green">{value}</span></p></div>}
                </div>
            </div>
        );
    }
}

export default HouseForm;