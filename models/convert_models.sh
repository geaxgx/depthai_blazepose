# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

source /opt/intel/openvino_2021/bin/setupvars.sh

convert_model () {
	model_name=$1
	if [ -z "$2" ]
	then
		arg_mean_values=""
	else
		arg_mean_values="--mean_values ${2}"
	fi
	if [ -z "$3" ]
	then
		arg_scale_values=""
	else
		arg_scale_values="--scale_values ${3}"
	fi
	mean_values=$2
	scale_values=$3
	tflite2tensorflow \
		--model_path ${model_name}.tflite \
		--model_output_path ${model_name} \
		--flatc_path ../../flatc \
		--schema_path ../../schema.fbs \
		--output_pb True
	# Generate Openvino "non normalized input" models: the normalization has to be mode explictly in the code
	#tflite2tensorflow \
	#  --model_path ${model_name}.tflite \
	#  --model_output_path ${model_name} \
	#  --flatc_path ../../flatc \
	#  --schema_path ../../schema.fbs \
	#  --output_openvino_and_myriad True 
	# Generate Openvino "normalized input" models 
	/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py \
		--input_model ${model_name}/model_float32.pb \
		--model_name ${model_name} \
		--data_type FP16 \
		${arg_mean_values} \
		${arg_scale_values} \
		--reverse_input_channels
	/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile \
		-m ${model_name}.xml \
		-ip u8 \
		-VPU_NUMBER_OF_SHAVES 4 \
		-VPU_NUMBER_OF_CMX_SLICES 4 \
		-o ${model_name}.blob
}

convert_model pose_detection "[127.5,127.5,127.5]"  "[127.5,127.5,127.5]"
convert_model pose_landmark_full_body "" "[255.0,255.0,255.0]"
convert_model pose_landmark_upper_body "" "[255.0,255.0,255.0]"


