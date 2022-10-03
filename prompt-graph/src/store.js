// This is the store for the User Interface; it'll allow for easier communication between different components
import React from "react"
import { extendObservable } from "mobx"
import * as d3 from 'd3';
import { generateConfig } from "./utils"

// Defining the BackendStore
class BackendStore {

    // This is the constructor; it'll define different attributes that're going to be monitored by MobX
    constructor(state = {}) {

        const color_scheme = d3.schemeSet1

        window.addEventListener('resize', ()=>{
            this.size_change += 1
        });

        // MobX's extendObservable method will make sure to propagate changes to the aforementioned attributes throughout the 
        // different components that use them 
        extendObservable(
            this,
            {

                size_change: 0,

                // The list of prompts will determine which buttons are created 
                prompts: [
                    { "type": "text", "prompt": "", "key": 0, "color": color_scheme[0], "points": [], "selected": null}
                ],

                // This is the index of the "prompts" array that represents the prompt we're currently editing in the curve editor
                active_prompt_key: 0,

                running_prompt_count: 1,
                prompt_key_to_idx: { 0: 0 },
                color_scale: color_scheme,

                // This controls how many frames will be shown in the bottom 
                frame_amt: 30,

                // This indicates how many frames per second the resulting video will be 
                frames_per_second: 6,

                // This stores the name of the configuration file 
                config_file_name: "Prompt Graph Output",

                prompt_key_to_path_node: {},

                // Whenever this integer increases, we'll store the current paths in the prompt_key_to_path_node Object
                config_trigger: 0,

                // This Object contains a mapping of prompt_keys to "coordMethods", which will convert a pair
                // of a screen XY coordinates to "graph space" coordinates
                prompt_key_to_coordMethod: {},

                // This is similar
                prompt_key_to_xGraphSpaceCoordMethod: {},

                // This defines the image width and image height
                image_width: 512,
                image_height: 512,
                image_seed: 1,
                diffusion_steps: 50

            }, state
        )
    }

    // This method will add a new Text Prompt to the prompts list 
    addTextPrompt() {
        this.prompts.push({
            "type": "text",
            "prompt": "",
            "key": this.running_prompt_count,
            "color": this.color_scale[this.running_prompt_count % this.color_scale.length],
            "points": []
        })
        this.prompt_key_to_idx[this.running_prompt_count] = this.prompts.length - 1
        this.changeActivePromptKey(this.prompts.length - 1)
        this.incrementRunningPromptCount()
    }

    // This method will add a new Image Prompt to the prompts list
    addImagePrompt() {
        this.prompts.push({ "type": "image", "prompt": "", "key": this.running_prompt_count, "points": [] })
        this.prompt_key_to_idx[this.running_prompt_count] = this.prompts.length - 1
        this.incrementRunningPromptCount()
    }

    // This method will change the index of the prompt we're currently editing in the curve editor
    changeActivePromptKey(new_key) {
        console.log(`active_prompt_key is ${new_key}`)
        this.active_prompt_key = new_key
    }

    // This method will remove the prompt that's at the given index
    removePromptWithKey(key_to_remove) {
        this.prompts.splice(this.prompt_key_to_idx[key_to_remove], 1)
        delete this.prompt_key_to_idx.key_to_remove
        this.incrementRunningPromptCount()
        this.refreshPromptToIdx()
    }

    incrementRunningPromptCount() {
        this.running_prompt_count = this.running_prompt_count + 1
    }

    refreshPromptToIdx() {
        var new_prompt_to_idx = {}
        this.prompts.forEach((prompt_dict, idx) => {
            new_prompt_to_idx[prompt_dict["key"]] = idx
        })
        this.prompt_key_to_idx = new_prompt_to_idx
    }

    configButtonPress() {
        this.config_trigger++
        generateConfig(
            this.prompt_key_to_path_node,
            this.prompt_key_to_coordMethod,
            this.frame_amt,
            this.frames_per_second,
            this.prompts,
            this.prompt_key_to_xGraphSpaceCoordMethod,
            this.config_file_name,
            this.image_seed,
            this.diffusion_steps,
            this.image_width,
            this.image_height
        )
    }

}

// Expose a BackendStoreContext for use in other components 
export const BackendStoreContext = React.createContext(new BackendStore())

