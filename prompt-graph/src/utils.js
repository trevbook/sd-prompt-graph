import $ from 'jquery';

// When given an X value and an SVG path, this function will find the Y value. 
// Taken from the following StackOverflow answer: https://stackoverflow.com/a/17896375
export function findYatXbyBisection(x, path, error = 0.01) {

    try {
        var length_end = path.getTotalLength()
            , length_start = 0
            , point = path.getPointAtLength((length_end + length_start) / 2) // get the middle point
            , bisection_iterations_max = 50
            , bisection_iterations = 0

        error = error || 0.01

        while (x < point.x - error || x > point.x + error) {
            // get the middle point
            point = path.getPointAtLength((length_end + length_start) / 2)

            if (x < point.x) {
                length_end = (length_start + length_end) / 2
            } else {
                length_start = (length_start + length_end) / 2
            }

            // Increase iteration
            if (bisection_iterations_max < ++bisection_iterations)
                break;
        }
        return point.y
    }

    catch {
        return null
    }
}

// Snagged this method from: https://stackoverflow.com/a/51215842
function exportToJson(objectData, file_name) {
    let filename = `${file_name}.json`;
    let contentType = "application/json;charset=utf-8;";
    if (window.navigator && window.navigator.msSaveOrOpenBlob) {
      var blob = new Blob([decodeURIComponent(encodeURI(JSON.stringify(objectData, null, 2)))], { type: contentType });
      navigator.msSaveOrOpenBlob(blob, filename);
    } else {
      var a = document.createElement('a');
      a.download = filename;
      a.href = 'data:' + contentType + ',' + encodeURIComponent(JSON.stringify(objectData, null, 2));
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }

// This function will generate a configuration file based on the 
export function generateConfig(prompt_key_to_path_node, prompt_key_to_coordMethod, frame_amt,
    frames_per_second, prompts, prompt_key_to_xGraphSpaceCoordMethod, filename,
    seed, steps, width, height) {

    // Declare the prompt configuration Object
    var config = {
        "frame_amt": frame_amt,
        "fps": frames_per_second,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height
    }

    // Add the different prompts to the config
    config["prompts"] = prompts

    // Iterate through each of the different frames and calculate prompt strength at that frame
    config["prompt_strength_by_frame"] = {}
    var prompt_key_to_initial_x_coord = {}
    for (var frame = 0; frame < frame_amt; frame++) {
        config["prompt_strength_by_frame"][frame] = {}

        // Iterate through each of the different prompts and calculate their strength at this frame 
        for (const cur_prompt_info of prompts) {

            const cur_key = cur_prompt_info.key

            const cur_prompt_path = prompt_key_to_path_node[cur_key]
            const cur_prompt_path_className = cur_prompt_path.getAttribute("class")
            const cur_prompt_path_node = $(`.${cur_prompt_path_className}`)[0]
            const cur_prompt_bounds = cur_prompt_path_node.getBoundingClientRect()

            const cur_frame_x_coord_space = prompt_key_to_xGraphSpaceCoordMethod[cur_key](frame)
            var graph_space_y = null
            if (cur_frame_x_coord_space >= cur_prompt_bounds.left && cur_frame_x_coord_space <= cur_prompt_bounds.right) {
                const cur_frame_y_coord_space = findYatXbyBisection(cur_frame_x_coord_space, cur_prompt_path)
                graph_space_y = prompt_key_to_coordMethod[cur_key]([cur_frame_x_coord_space, cur_frame_y_coord_space])[1]
            }


            config["prompt_strength_by_frame"][frame][cur_key] = graph_space_y
        }

    }

    // Add the filename to the config
    config["filename"] = filename

    // Log the config to the console (for testing purposes)
    exportToJson(config, filename)
}