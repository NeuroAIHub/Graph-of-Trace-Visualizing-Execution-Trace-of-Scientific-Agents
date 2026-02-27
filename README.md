# TraceGraph
![TraceGraph](./figures/UI.png "UI Example of TraceGraph")
Our online demo is available [here](http://8.145.42.208:8080 "TraceGraph Online Demo")
## Quick Start
- Follow the demo link above, and you'll see a modified [OpenHands](https://docs.openhands.dev/overview/introduction) UI like this
![openhands_init](./figures/openhands_init.jpeg)
- Click on the user icon in the bottom-left corner to go to Settings. Then, select Large Language Model​ and enter your corresponding API key.
![openhands_setting](./figures/openhands_setting.jpeg)
- Go back to initial panel by clicking openhands icon in the top-left corner, and start a new conversation.
- You can find TraceGraph (Graph of Thought) on the right side.
![openhands_got](./figures/openhands_got.jpeg)
Here is a prompt template you can use when interacting with the agent:
```
Please base your analysis on the xxxx dataset, complete the following research tasks under a consistent experimental and evaluation framework, and record your trajectory using build_trace tool:
### Background and motivation：

### Research Goals：

```

## Notes
- You can adjust the layout​ by dragging​ and zooming on the canvas.
- Click on a node to view its details.
