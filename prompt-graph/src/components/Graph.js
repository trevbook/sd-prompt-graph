// Different import statements for the Graph component 
import { useD3 } from '../hooks/useD3';
import React, { useContext, useDebugValue, useEffect } from 'react';
import * as d3 from 'd3';
import $ from 'jquery';
import PropTypes from 'prop-types';
import { observer } from 'mobx-react';
import { BackendStoreContext } from "../store"
import Curve from "./Curve"
import "../css/responsive-svg.css"

// Declare the Graph component, and make sure to use a MobX observer
const Graph = observer((props) => {

  // Set up the "points" array
  var points = []

  // Indicate that we're using the BackendStoreContext 
  const backend = useContext(BackendStoreContext)

  // Set up the SVG that we're going to be using 
  const ref = useD3(
    (svg) => {

      svg.select(".x-axis-grid").selectAll("*").remove()
      svg.select(".y-axis-grid").selectAll("*").remove()
      svg.select(".x-axis").selectAll("*").remove()
      svg.select(".y-axis").selectAll("*").remove()

      // =======================
      //      GRAPH SETUP
      // =======================
      // Below, I'm going to set up the graph itself by creating an X and Y axis 

      // Determining the width and height of the SVG that contains the graph
      const graph_width = $(".main-svg")[0].getBoundingClientRect().width
      const graph_height = $(".main-svg")[0].getBoundingClientRect().height

      // We're also going to indicate the maxmimum value for the y-axis
      const y_axis_max_val = 2

      // Declaring a scale for both of the axes
      const x_axis_scale = d3.scaleLinear([0, backend.frame_amt], [0, graph_width - (props.margin * 2)])
      const y_axis_scale = d3.scaleLinear([y_axis_max_val, 0], [0, graph_height - (props.margin * 2)])

      // Declaring the amount of ticks on the y-axis 
      const y_axis_tick_amt = 5

      // Setting up the X-axis of the graph
      svg.select(".x-axis").call(
        d3.axisBottom(x_axis_scale)
      ).attr("transform", `translate(${props.margin}, ${graph_height - (props.margin)})`)

      svg.select(".x-axis-grid").call(
        d3.axisBottom(x_axis_scale).tickSize(-(graph_height - (props.margin * 2))).tickFormat((val) => { return "" })
      ).attr("transform", `translate(${props.margin}, ${graph_height - (props.margin)})`)
      svg.select(".x-axis-grid").selectAll("line").style("stroke", "#ededed")

      // Adding two black lines to make a boundary box for the graph
      svg.select(".x-axis").append("line").style("stroke", "black").attr("x1", graph_width - (props.margin * 2)).attr("x2", graph_width - (props.margin * 2))
        .attr("y1", 0).attr("y2", -1 * (graph_height - (props.margin * 2)))
      svg.select(".x-axis").append("line").style("stroke", "black").attr("x1", 0).attr("x2", graph_width - (props.margin * 2))
        .attr("y1", -1 * (graph_height - (props.margin * 2))).attr("y2", -1 * (graph_height - (props.margin * 2)))

      // Setting up the Y-axis of the graph
      svg.select(".y-axis").call(
        d3.axisLeft(y_axis_scale).ticks(y_axis_tick_amt)
      ).attr("transform", `translate(${props.margin}, ${props.margin})`)

      // Setting up the Y-axis of the graph
      svg.select(".y-axis-grid").call(
        d3.axisLeft(y_axis_scale).tickSize(-(graph_width - (props.margin * 2))).tickFormat((val) => { return "" }).ticks(y_axis_tick_amt)
      ).attr("transform", `translate(${props.margin}, ${props.margin})`)
      svg.select(".y-axis-grid").selectAll("line").style("stroke", "#ededed")

    }, [JSON.stringify(points), window.height, window.width, backend.frame_amt],
  );

  return (
    <div id={"chart_container"} style={{"height": 600}}>
      <svg
        ref={ref}
        style={{
          height: "100%",
          width: "100%",
        }}
        className="main-svg"
      >



        <g className="x-axis-grid" />
        <g className="y-axis-grid" />
        <g className="x-axis" />
        <g className="y-axis" />

        {backend.prompts.map((prompt_info, idx) => {
        return (
          <Curve prompt_key={prompt_info.key} prompt_color={prompt_info.color} points={prompt_info.points} key={prompt_info.key} selected={prompt_info.selected}> 
          </Curve>
        )
      })}


      </svg>
      
    </div>
  );
}
)

// Indicating the types of various props for this component
Graph.propTypes = {
  margin: PropTypes.number
}

// Setting up some default props for this component
Graph.defaultProps = {
  margin: 30
}

// Export this component for use in other Components 
export default Graph;