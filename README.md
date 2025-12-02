# How to start
### Recreate the stack (Docker Compose):

```shell
make start
``` 
This command with start 4 services:
* dashboard
* api
* zeek network sniffer
* zeek to api stream

Zeek network sniffer attaches to the `eth0` interface by default, if your network setup is different - update `docker-compose.yaml` accordingly.

To stop the stack run
```shell
make stop
```
 
### Streemlit Dashboard:
[Streamlit Dashboard](http://localhost:8501/?clear_cache=1)


