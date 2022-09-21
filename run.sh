#!/usr/bin/env bash

########################################################################################################################

function run_fedavg() {
	pushd "${path}"/src/ > /dev/null 2>&1 || exit
	${command}
	popd > /dev/null 2>&1 || exit
}

path=$(pwd)
port=$(python get_free_port.py)
config_path="$1"
#command="python3 -W ignore ${profiler} run.py"
command="python3 -W ignore -m torch.distributed.launch --nproc_per_node * --master_port=${port} run.py"

while IFS="=" read -r arg value; do

  if [ "${arg}" != "" ]; then
    if [ "${value}" = "" ]; then
      command="${command} --${arg}"
    else
      declare "${arg}"="${value}"
      if [ "${arg}" = "device_ids" ]; then
        device_ids="${value}"
        device_ids="${device_ids:1:-1}"
        IFS=' ' read -r -a device_ids_array <<< "${device_ids}"
        num_devices=${#device_ids_array[@]}
        command="${command} --device_ids ${device_ids}"
      elif [ "${arg}" = "batch_size" ]; then
        batch_size_per_device=$((batch_size/num_devices))
        command="${command} --batch_size ${batch_size_per_device%.*}"
      else
        command="${command} --${arg} ${value}"
      fi
    fi
  fi

done < "$config_path"

batch_size_per_device=$((batch_size/num_devices))
command=${command//[*]/${num_devices}}

echo "GPUs in usage:" "${device_ids_array[@]}"

pushd ../ > /dev/null 2>&1 # hide output: > /dev/null 2>&1

if [ ! -d "${path}/data/" ] || [ ! -d "${path}/src/" ]; then
	echo "Couldn't find data/ and/or src/ directories"
fi

if [ "${framework}" = "federated" ]; then
	echo "Running ${algorithm} experiment with ${num_rounds} rounds, ${num_epochs} local epochs and ${clients_per_round} clients per round"
else
	echo "Running Centralized experiment with ${num_epochs} epochs"
fi

run_fedavg "${clients_per_round}" "${num_epochs}"

popd ../ > /dev/null 2>&1
