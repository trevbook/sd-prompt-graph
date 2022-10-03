// This file contains the code for the Control Panel
import { useD3 } from '../hooks/useD3';
import React, { useContext, useDebugValue, useEffect } from 'react';
import * as d3 from 'd3';
import $ from 'jquery';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import { Grid, Box, TextField, Button, Tooltip } from '@mui/material';


// This is the Control Panel
const ControlPanel = observer((props) => {

    // Indicate that we're using the BackendStoreContext 
    const backend = useContext(BackendStoreContext)

    // Determine what the ControlPanel will return
    return (
        <div style={{ "margin": "20px" }}>

            <h1>
                Control Panel
            </h1>

            <div style={{"marginTop": "35px"}}>
                <Grid container spacing={4}>
                    <Grid item xs={6}>
                        
                        {/* This is the input for how wide the resulting images ought to be */}
                        <div>
                            <Tooltip title="How wide the resulting images are" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="image_width_input"
                                    label="Image Width"
                                    defaultValue={backend.image_width}
                                    type="number"
                                    onChange={(event) => {
                                        backend.image_width = Math.ceil(event.target.value/64)*64
                                    }}
                                />
                            </Tooltip>
                        </div>

                    </Grid>
                    <Grid item xs={6}>
                        
                        {/* This is the input for how tall the resulting images ought to be */}
                        <div>
                            <Tooltip title="How tall the resulting images are" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="image_height_input"
                                    label="Image Height"
                                    defaultValue={backend.image_height}
                                    type="number"
                                    onChange={(event) => {
                                        backend.image_height = Math.ceil(event.target.value/64)*64
                                    }}
                                />
                            </Tooltip>
                        </div>

                    </Grid>


                </Grid>
            </div>

            <div style={{"marginTop": "35px"}}>
                <Grid container spacing={4}>
                    <Grid item xs={6}>
                        
                        {/* This is the input for the seed used */}
                        <div>
                            <Tooltip title="The seed used for the StableDiffusion generation" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="image_seed"
                                    label="Image Seed"
                                    defaultValue={backend.image_seed}
                                    type="number"
                                    onChange={(event) => {
                                        backend.image_seed = Number(event.target.value)
                                    }}
                                />
                            </Tooltip>
                        </div>

                    </Grid>
                    <Grid item xs={6}>
                        
                        {/* This is the input for the steps run */}
                        <div>
                            <Tooltip title="How many diffusion steps will run per image" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="image_steps"
                                    label="Diffusion Steps"
                                    defaultValue={backend.diffusion_steps}
                                    type="number"
                                    onChange={(event) => {
                                        backend.diffusion_steps = event.target.value
                                    }}
                                />
                            </Tooltip>
                        </div>

                    </Grid>


                </Grid>
            </div>

            <div style={{"marginTop": "35px"}}>
                <Grid container spacing={4}>
                    <Grid item xs={6}>
                        {/* This is the input for how many frames we're creating */}
                        <div>
                            <Tooltip title="The number of frames your animation will have" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="total_frames_input"
                                    label="Total Frames"
                                    defaultValue={backend.frame_amt}
                                    type="number"
                                    onChange={(event) => {
                                        backend.frame_amt = event.target.value
                                    }}
                                />
                            </Tooltip>
                        </div>

                    </Grid>
                    <Grid item xs={6}>
                        {/* This is the input for how many frames per second the resulting video will be */}
                        <div>
                            <Tooltip title="How many frames are shown in a second" placement="top"
                                enterDelay={200} enterNextDelay={200}>
                                <TextField
                                    id="fps_input"
                                    label="Frames Per Second"
                                    defaultValue={backend.frames_per_second}
                                    type="number"
                                    onChange={(event) => {
                                        backend.frames_per_second = event.target.value
                                    }}
                                />
                            </Tooltip>
                        </div>
                    </Grid>


                </Grid>
            </div>

            {/* This Div will indicate the resulting length of the video */}
            <div style={{ "marginTop": "10px" }}>
                <b>Resulting Video Length:</b>
                <div style={{ "display": "inline-block", "marginLeft": "0.5em" }}>
                    {(backend.frame_amt / backend.frames_per_second).toFixed(2)} seconds
                </div>
            </div>

            <div style={{ "marginTop": "35px" }}>
                <Tooltip title="The name of the configuration file that will be created" placement="top"
                    enterDelay={200} enterNextDelay={200}>
                    <TextField
                        id="name_input"
                        label="Config File Name"
                        defaultValue={backend.config_file_name}
                        type="text"
                        fullWidth
                        onChange={(event) => {
                            backend.config_file_name = event.target.value
                        }}
                    />
                </Tooltip>
            </div>



            {/* This Div will create a LARGE "generate" button */}
            <div style={{ "marginTop": "45px", "height": "40px" }}>
                <Button
                    variant="contained"
                    style={{ "width": "100%", "height": "100%" }}
                    onClick={() => {
                        backend.configButtonPress()
                    }}
                >
                    Generate Configuration File
                </Button>
            </div>


        </div >
    )

})

// Export the ControlPanel for use in other components
export default ControlPanel