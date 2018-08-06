import React, { Component } from 'react';
import logo from './logo.svg';
import assistant_logo from './assets/assistant_logo.png';
import './App.css';
import HouseForm from './components/HouseForm';

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          {/* <img src={logo} className="App-logo" alt="logo" /> */}
          <img src={assistant_logo} alt="Acumos Property Assistant" />
          {/* <h1 className="App-title">Acumos Property Assistant</h1> */}
        </header>
        <div className="App-intro">
          <HouseForm/>
        </div>
      </div>
    );
  }
}

export default App;
