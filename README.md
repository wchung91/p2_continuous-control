#Project Name: Continuous Control
#1. Description of Environment
We train a 2 DOF robotic arm to reach a particular target in the field. The reward is the sum of reward_dist and reward_ctrl. reward_dist is the distance between the fingertip of the robot and the target. reward_ctrl is the sum of squared actions. The state space is 33 dimensions. The action space is 2 dimensions and the values can be between -1 and 1. The action controls the joints of the robot. The problem is considered solved if the average reward is at least 30+ over 100 episodes. 


#2. Description of Installation 
We use a docker with the nvidia driver and isolate the environment. Inside the docker, we then create a virtual environment to use Python 3.6. In the virtual environment, we install pytorch and unityagents. 

#3. Installation Guide 
3.1 This installation guide assumes that 
     -OS is Ubuntu 16.04. 
     -Docker is installed  (https://docs.docker.com/engine/install/ubuntu/) 
     -Nvidia driver is installed Cuda is installed 
     -Cudnn is installed 
     -nvidia-docker is installed (https://github.com/NVIDIA/nvidia-docker) 
     -git is installed 

3.2 Clone the repository 

   git clone https://github.com/wchung91/p2_continuous-control.git

3.3 Build the dockerfile. Run the command below in the terminal and it will create an image named rl_env.

   sudo docker build . --rm -t rl_env_cont

If you use a different GPU from RTX 2080, you need to change the dockerfile. Open the dockerfile and change “pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime” to the docker image corresponding to the cuda and cudnn installed on your computer. 

3.4 Create a docker container from the docker image, but you need to change “/home/wally-server/Documents/p2_continuous-control” in the command below and then run the command. “/home/wally-server/Documents/p2_continuous-control” is the directory of the volume. You should change “/home/wally-server/Documents/p2_continuous-control” to the path to the cloned repository. That way the docker container has access to the files cloned from the repository. One you changed the command, run the command. 

   sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p2_continuous-control:/workspace --name p2_continuous-control_container rl_env_cont /bin/bash

3.5 To access the container run,  
 
   sudo docker exec -it p2_continuous-control_container /bin/bash

3.6 Inside the container run the command below to initialize conda with bash 

   conda init bash

3.7 You need to close and reopen a new terminal. You can do that with the command from 3.5. Create a virtual environment named “p2_env” with python 3.6 with the following code 

   conda create -n p2_env -y python=3.6

3.8 Activate the environment “p2_env” with the command below. 

   conda activate p2_env

3.9 Inside the virtual environment, install pytorch with the command below. You’ll have to install the correct pytorch version depending on your cuda and cudnn version. 

   pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

3.10 Install unityagents with the following code. 

   pip install unityagents

3.11 Download the unity environments with the following commands. Since we are using a docker, you’ll have to use Banana_Linux_NoVis because no display is available. 

   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip && unzip Reacher_Linux_NoVis.zip


3.12 To run the training code, go to main.py and set “Train = 1”. Then, run the command below

   python main_one.py 

The code will print the average scores, and it will create a figure called “ScoresTraining.png”

3.13 To run the testing code, go to main.py and set “Train = 0”. Then, run the command below 

   python main_one.py

The code will print the average scores, and it will create a figure called “TestScores.png”

#4. About the Code 
main_one.py - contains the main method for running the code. The code is divided into training code and testing code 
ddpg_agent.py - contains the code for the ddpg_agent and experience replay buffer.
model.py - contains the actor and critic models. 
checkpoint_actor.pth - trained actor model
checkpoint_critic.pth - trained critic model

#5. Incase code doesn't run 
Make sure in main_one.py the code below points to the right path of the "Reacher.x86_64"

   env = UnityEnvironment(file_name='./Reacher_Linux_NoVis/Reacher.x86_64')




