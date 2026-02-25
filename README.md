# TraceGraph
mcp server: "TraceGraph"启动方式如下：
- tmux attach -t mcp_server_running_here
- conda activate mcp_env 
- cd /srv/MCP_tools
- fastmcp run server.py: fastmcp run server.py:mcp --transport http --host 0.0.0.0 --port 8000
Openhands中配置：
shttp
http://host.docker.internal:8000/mcp


