// This file contains the code for a new Text Prompt
import React, { useContext } from 'react';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import { TextareaAutosize, TextField } from '@mui/material';
import { Grid, Box } from '@mui/material';
import TextPrompt from './TextPrompt';
import AddPromptButton from './AddPromptButton';

// Declare the PromptArea 
const PromptArea = observer((props) => {

    // Indicate that we're using the BackendStoreContext 
    const backend = useContext(BackendStoreContext)

    // Make a return statement for this component 
    return (
        <div>
            <Grid container spacing={2} direction={"row"} justify={"flex-start"} alignItems={"flex-start"}>
                {backend.prompts.map((elem, idx) => {
                    if (elem.type == "text") {
                        return (
                            <Grid item xs={2} key={elem.key}>
                                <div>
                                    <TextPrompt prompts_idx={elem.key} prompt_key={elem.key} prompt={elem.prompt}></TextPrompt>
                                </div>
                            </Grid>
                        )
                    }
                })}
                <Grid item xs={2}>
                    <div>
                        <AddPromptButton></AddPromptButton>
                    </div>
                </Grid>
            </Grid>
        </div>
    )

})

// Export the PromptArea 
export default PromptArea