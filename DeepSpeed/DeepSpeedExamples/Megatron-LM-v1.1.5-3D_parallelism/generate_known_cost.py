# Generate the profile cost by launching configurations with different model
# paralleism degree
import os
import subprocess
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str, help="Which experiment settings, possible choices are:\
                   homogeneous, het_cluster, het_model")

args = parser.parse_args()

# home_dir = os.environ['HOME']
home_dir = "/root/AMP/AMP"
workdir_path = os.path.join(home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
example_path = os.path.join(workdir_path, "examples")

assert args.setting in ["homogeneous", "het_cluster", "het_model"], "possible settings are: homogeneous, het_cluster, het_model" 

# 这里也要修改
if args.setting == "homogeneous":
    # profile_path = os.path.join(example_path, "ds_pretrain_gpt2_profile.sh")
    # profile_args = f" 24 1024 profile "
    profile_path = os.path.join(example_path, "ds_pretrain_gpt3_profile.sh")
    profile_args = f" 30 4096 profile " # layers hidden_size type
    # record_path = os.path.join(workdir_path, "known_cost", "gpt2_G4_")
    record_path = os.path.join(workdir_path, "known_cost", "gpt3_G4_")
elif args.setting == "het_cluster":
    profile_path = os.path.join(example_path, "ds_pretrain_gpt2_profile.sh")
    profile_args = f" 24 1024 profile "
    record_path = os.path.join(workdir_path, "known_cost", "gpt2_P3_")
else:
    profile_path = os.path.join(example_path, "ds_pretrain_transgan_profile.sh")
    profile_args = f" 1024 profile "
    record_path = os.path.join(workdir_path, "known_cost", "transgan_P3_")

json_path = "examples/ds_config.json"
with open(json_path) as json_file:
    dict = json.load(json_file)
    dict["train_micro_batch_size_per_gpu"] = 1
    dict["gradient_accumulation_steps"] = 1
with open(json_path, 'w') as outfile:
    json.dump(dict, outfile)

# Saving the profiling result to record_path
# for mp_degree in [1,2,4]: # tp_degree
for mp_degree in [1,2,4,8]: 
    cur_record_path = record_path + str(mp_degree)
    os.environ["amp_record_path"] = cur_record_path
    cur_profile_args = profile_args + f"{mp_degree}"
    cmd = "bash " + profile_path  + cur_profile_args
    subprocess.run(cmd, shell=True)
