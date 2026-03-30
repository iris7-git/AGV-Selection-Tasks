# How to run

## Build
docker build -t task1subtask2 .

## Run (Linux with GUI)
xhost +local:docker

docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  task1subtask2