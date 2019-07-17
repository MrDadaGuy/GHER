from gym.envs.registration import register

register(
    id='Ros-Unity-Sim-v1',
    entry_point='GHER.gmgym.ros_unity_env:RosUnityEnv',
)
