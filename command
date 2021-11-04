sudo docker build . --rm -t rl_env_cont

sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p2_continuous-control:/workspace --name p2_continuous-control_container rl_env_cont /bin/bash

sudo docker exec -it p2_continuous-control_container /bin/bash


sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p2_continuous-control:/workspace --name test rl_env /bin/bash

sudo docker exec -it p2_continuous-control_container /bin/bash
