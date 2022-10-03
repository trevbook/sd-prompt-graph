// Different import statements for the Curve component 
import { useD3 } from '../hooks/useD3';
import React, { useContext, useDebugValue, useEffect } from 'react';
import * as d3 from 'd3';
import $ from 'jquery';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import { findYatXbyBisection } from "./../utils"
import { usePreviousProps } from '@mui/utils';

// Declare the Curve component, and make sure to use a MobX observer
const Curve = observer((props) => {

    // Indicate that we're using the BackendStoreContext 
    const backend = useContext(BackendStoreContext)
    var selected = null

    const ref = useD3((curve_editor) => {

        console.log("hey curve")

        // Determining the width and height of the SVG that contains the graph
        const graph_width = $(".main-svg")[0].getBoundingClientRect().width
        const graph_height = $(".main-svg")[0].getBoundingClientRect().height

        console.log(`graph width: ${graph_width}`)

        // We're also going to indicate the maxmimum value for the y-axis
        const y_axis_max_val = 2

        // Declaring a scale for both of the axes
        const x_axis_scale = d3.scaleLinear([0, backend.frame_amt], [0, graph_width - (props.margin * 2)])
        const y_axis_scale = d3.scaleLinear([y_axis_max_val, 0], [0, graph_height - (props.margin * 2)])

        // Declaring the amount of ticks on the y-axis 
        const y_axis_tick_amt = 5

        // =======================
        //      CURVE EDITOR  
        // =======================
        // The code below is directly relevant to the behavior of the curve editing functionality 

        // This indicates that the "keydown()" method ought to be run if there's a key pressed
        if (backend.active_prompt_key == props.prompt_key) {
            d3.select(window)
                .on("keydown", keydown);
        }

        // Here, we'll define the "props.points" list that will store the different props.points for the line 
        var prevSelected = null

        // Grab the area of the SVG that corresponds to the "plot area" 
        curve_editor.select(`.curve-rec-${props.prompt_key}`)
            .attr("fill", "none")
            .attr("width", graph_width)
            .attr("height", graph_height).call(update_visualization);

        // Next, we'll define a couple of functions that will help with the creation of new lines in the curve editor 
        curve_editor.attr("pointer-events", props.prompt_key == backend.active_prompt_key ? "all" : "none").call(
            d3.drag().subject(determine_drag_subject).on("start", drag_started).on("drag", dragging)
        )

        // This function will update the visualization accordingly 
        function update_visualization() {

            const curve = d3.line().curve(d3.curveLinear);

            // This statement will indicate that the "path" must be drawn through a particular line function (in this case, a linear curve)
            if (!curve_editor.select(`.curve-path-${props.prompt_key}`).empty()) {
                curve_editor.select(`.curve-path-${props.prompt_key}`).attr("d", curve(props.points)).attr("id", "current_curve").attr("fill", "none").attr("stroke", props.prompt_color).attr("stroke-width",
                    props.prompt_key == backend.active_prompt_key ? 5 : 2);
            }

            // Grab the current path, and set this in the 
            const my_path = d3.select(`.curve-path-${props.prompt_key}`).node()
            backend.prompt_key_to_path_node[props.prompt_key] = my_path

            // This code effectively creates the new circle whenever a particular element enters the "props.points" array. A lot of the 
            const circle = curve_editor.selectAll("g")
                .data(props.points, d => d)
            circle.enter().append("g")
                .call(g => g.append("circle")
                    .attr("r", 20)
                    .attr("fill", "none"))
                .call(g => g.append("circle")
                    .attr("r", 0)
                    .attr("stroke", "black")
                    .attr("stroke-width", 1.5)
                    .transition()
                    .duration(750)
                    .ease(d3.easeElastic)
                    .attr("r", 5))
                .merge(circle) // This flattens things into a single array
                .attr("transform", d => `translate(${d})`) // This moves all of the different circles to where they ought to be based on their props.points
                .select("circle:last-child")
                .attr("fill", d => d === selected ? "lightblue" : "black"); // This changes the color of the point to indicate which is selected


            circle.exit().remove(); // When a member of the "props.points" array is removed, we'll remove the <g> element corresponding to it 


        }

        // This function will determine which point the user is currently dragging
        function determine_drag_subject(event) {

            // Grab the data of the event target 
            let target_data = event.sourceEvent.target.__data__

            // If there is no target data, then we're going to add a new point w/ this data
            if (!target_data) {

                target_data = [event.x, event.y]
                selected = target_data

                // We're going to figure out where in the target_data this new point belongs
                var idx_to_insert = props.points.length
                props.points.some((point, idx) => {
                    if (point[0] > target_data[0]) {
                        idx_to_insert = idx
                        return true
                    }
                })
                props.points.splice(idx_to_insert, 0, target_data)
                update_visualization()
            }

            // Now, we're going to return the target_data
            return target_data
        }

        // This function will be run when the dragging behavior starts
        function drag_started({ subject }) {
            selected = subject
            update_visualization()
        }

        // This function will be run as the dragging behavior is happening; essentially, it'll update the coordinates of the 
        // point that's currently being dragged around 
        function dragging(event) {


            // Figure out the two props.points next to this one
            const selected_index = props.points.indexOf(selected)
            var lower_x_bound = 30
            var upper_x_bound = Math.min(graph_width - (props.margin), event.x)
            if (props.points.length - 1 > selected_index) {
                upper_x_bound = props.points[selected_index + 1][0]
            }
            if (selected_index > 0) {
                lower_x_bound = props.points[selected_index - 1][0]
            }

            event.subject[0] = Math.max(lower_x_bound, Math.min(upper_x_bound, event.x)); // I'll have to experiment with this behavior 
            event.subject[1] = Math.max(30, Math.min((graph_height - props.margin), event.y));
            update_visualization();
        }

        // This method will convert an x-coordinate to "graph space"
        function xCoordToGraphSpace(x_client_pos) {
            const x_graph_pos = x_client_pos - props.margin
            const x_val = x_axis_scale.invert(x_graph_pos)
            return x_val
        }

        // This method will convert a y-coordinate to "graph space"
        function yCoordToGraphSpace(y_client_pos) {
            const y_graph_pos = (graph_height - props.margin) - y_client_pos
            const y_val = Math.abs(y_axis_scale.invert(y_graph_pos) - y_axis_max_val) * Math.sign(y_graph_pos)
            return y_val
        }

        function xGraphSpaceToCoord(x_graph_space) {
            return x_axis_scale(x_graph_space) + props.margin
        }

        // This method will convert a tuple of coordinates to "graph space"
        function coordsToGraphSpace(coords) {
            return [
                xCoordToGraphSpace(coords[0]),
                yCoordToGraphSpace(coords[1])
            ]
        }

        backend.prompt_key_to_coordMethod[props.prompt_key] = coordsToGraphSpace
        backend.prompt_key_to_xGraphSpaceCoordMethod[props.prompt_key] = xGraphSpaceToCoord

        // This is the function that'll be run whenever there's a key press
        function keydown(event) {

            // If there's nothing selected, we don't need to do anything
            if (!selected) {
                return
            };

            // If the key pressed is "Backspace" or "Delete", then we're going to remove the currently selected point
            // eslint-disable-next-line default-case
            switch (event.key) {
                case "-": {
                    event.preventDefault(); // "Don't take the default action if this event isn't handled"
                    const i = props.points.indexOf(selected);
                    if (i == -1) {
                        break
                    }
                    props.points.splice(i, 1);
                    selected = props.points.length ? props.points[i > 0 ? i - 1 : 0] : null; // This determines which point we ought to set as selected next
                    update_visualization();
                    break;
                }
            }
        }
    }, [props.prompt_key, backend.active_prompt_key, backend.frame_amt, backend.config_trigger])

    return (
        <g ref={ref} className="curveEditor" id={`curve_${props.prompt_key}`}>
            <rect className={`curve-rec-${props.prompt_key}`}></rect>
            <path className={`curve-path-${props.prompt_key}`}></path>
        </g>
    );
}
)

// Indicating the types of various props for this component
Curve.propTypes = {
    margin: PropTypes.number
}

// Setting up some default props for this component
Curve.defaultProps = {
    margin: 30
}

// Export this component for use in other Components 
export default Curve;