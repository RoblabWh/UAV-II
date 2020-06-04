input_dir="/home/gas/host/docker/struct2depth/images"
output_dir="/home/gas/host/docker/struct2depth/output"
model_checkpoint="/home/gas/host/docker/struct2depth/models/kitti/model-199160"

python3 inference.py \
    --logtostderr \
    --file_extension jpeg \
    --depth \
    --egomotion true \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint