// This file contains the code for a new Text Prompt
import React, { useContext } from 'react';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import { InputBase, TextareaAutosize, TextField } from '@mui/material';
import { Grid, Box } from '@mui/material';

// Declare the TextPrompt 
const TextPrompt = observer((props) => {

    // Indicate that we're using the BackendStoreContext 
    const backend = useContext(BackendStoreContext)

    // Grab the "color" associated with this prompt from the backend prompts list
    const prompt_color = backend.prompts[backend.prompt_key_to_idx[props.prompt_key]].color

    const border = `${props.prompt_key == backend.active_prompt_key ? 5 : 2}px solid ${prompt_color}`

    // Make a return statement for this component 
    return (
        <div
            onClick={() => {
                backend.changeActivePromptKey(props.prompt_key)
            }}
            style={{
                "width": "100%", "height": "max-content", "minHeight": 150, "position": "relative", "border": border,
                "borderRadius": "10px",
            }}>

            <Grid container spacing={2} style={{ }}>
                <Grid item xs={9}>
                    <div style={{ "height": "20%", "width": "100%", "marginLeft": "5px" }}>
                        <b>Text Prompt</b>
                    </div>
                </Grid>
                <Grid item xs={3}>
                    <div style={{ "textAlign": "end", "marginRight": "10px" }}>
                        <button onClick={() => {
                            backend.removePromptWithKey(props.prompt_key)
                        }}>
                            X
                        </button>
                    </div>
                </Grid>
            </Grid>


            <div style={{ "marginBottom": "10px", "position": "relative" }}>
                <InputBase fullWidth multiline rows={5} value={props.prompt}
                    inputProps={{
                        style: {
                            "textarea": {
                                "border": "none",
                                "outline": "none"
                            },
                            "marginLeft": "5px",
                            "marginLeft": "5px"

                        }
                    }}
                    style={{

                    }}
                    onChange={(event) => {
                        backend.prompts[backend.prompt_key_to_idx[props.prompt_key]].prompt = event.target.value
                    }}>
                </InputBase>
            </div>
        </div>
    )

})

// Export the TextPrompt 
export default TextPrompt