// This file contains the code for the "Add Prompt" button
import React, {useContext} from 'react';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import Button, { ButtonProps } from "@mui/material/Button"

// Declare the Add Prompt button
const AddPromptButton = observer((props) => {

    // Indicate that we're using the BackendStoreContext 
    const backend = useContext(BackendStoreContext)

    // This is what the AddPromptButton will render 
    return (

        // This Div contains both of the buttons to add a new Text/Image prompt 
        <div style={{ "width": "100%", "height": 160, "border": "1px dashed black", "position": "relative", "marginLeft": "10px"}}>
            <div style={{
                "margin": 0, "top": "50%", "position": "absolute", "transform": "translate(-50%, -50%)",
                "msTransform": "translateY(-50%)", "left": "50%", "width": "max-content"
            }}>
                <Button variant="outlined" onClick={()=>{backend.addTextPrompt()}}>
                    Text Prompt
                </Button>
            </div>
            {/* <div style={{
                "margin": 0, "top": "75%", "position": "absolute", "transform": "translate(-50%, -75%)",
                "msTransform": "translateY(-75%)", "left": "50%", "width": "max-content"
            }}>
                <Button variant="outlined" disabled onClick={()=>{backend.addImagePrompt()}}>
                    Image Prompt
                </Button>
            </div> */}
        </div>
    )
})

// Export this component for use in other components
export default AddPromptButton
