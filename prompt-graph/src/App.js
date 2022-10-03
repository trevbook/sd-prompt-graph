import logo from './logo.svg';
import './App.css';
import * as d3 from 'd3'
import Graph from "./components/Graph"
import AddPromptButton from './components/AddPromptButton';
import TextPrompt from './components/TextPrompt';
import PromptArea from './components/PromptArea';
import { Grid, Box } from '@mui/material';
import ControlPanel from './components/ControlPanel';
import {useState} from "react"

// This App is the 
function App() {

    document.title = 'Prompt Graph';

  return (
    <div style={{ "margin": "20px" }}>

      {/* This Box contains the Grids that make up the rest of the app */}
      <Box sx={{ flexGrow: 1 }}>

        {/* This Grid contains the Curve Editor and the Settings Pane */}
        <Grid container spacing={1} style={{"marginBottom": "20px"}}>
          <Grid item xs={9}>
            <div style={{"height": 600}}>
            <Graph/>
            </div>
          </Grid>

          {/* This is the Control Panel, where the different settings exist */}
          <Grid item xs={3}>
            <ControlPanel/>
          </Grid>

        </Grid>

        {/* This Grid contains the Prompt Area and the Generate button */}
        <Grid container spacing={1}>
          <Grid item xs={12}>
            <PromptArea></PromptArea>
          </Grid>
        </Grid>

      </Box>
    </div>
  );
}

export default App;
