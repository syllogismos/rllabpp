#!/bin/bash
{

die() { status=$1; shift; echo "FATAL: $*"; exit $status; }

EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"

aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value=CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01 --region us-west-2

aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=exp_prefix,Value=Anil First RL Experiment --region us-west-2

service docker start

docker --config /home/ubuntu/.docker pull dementrock/rllab3-shared

export AWS_DEFAULT_REGION=us-west-2

aws s3 cp loadsfad /tmp/rllab_code.tar.gz

mkdir -p /root/code/rllab

tar -zxvf /tmp/rllab_code.tar.gz -C /root/code/rllab

aws s3 cp --recursive <insert aws s3 bucket url for code>e/.mujoco/ /home/ubuntu/.mujoco

cd /root/code/rllab

aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value=CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01 --region us-west-2

mkdir -p /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1

aws s3 sync --exclude '*' --include '*.csv' --include '*.json' --include '*.pkl' --include '*.chk' --include '*.meta' <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1 /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1 --region us-west-2

while /bin/true; do
aws s3 sync --exclude '*'    --include '*.pkl'   --include '*.csv' --include '*.json' --include '*.chk' --include '*.meta' /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1 <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1
sleep 1200
done & echo sync initiated
while /bin/true; do
if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
then
logger "Running shutdown hook."
aws s3 cp /home/ubuntu/user_data.log <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1/stdout.log
aws s3 cp --recursive /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1 <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1
break
else
# Spot instance not yet marked for termination.
sleep 5
fi
done & echo log sync initiated

docker run -e "RLLAB_USE_GPU=False" -e "AWS_ACCESS_KEY_ID=<insert aws key>" -e "AWS_SECRET_ACCESS_KEY=<insert aws secret>" -v /home/ubuntu/.mujoco:/root/.mujoco -v /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1:/tmp/expt -v /root/code/rllab:/root/code/rllab -i dementrock/rllab3-shared /bin/bash -c 'echo "Running in docker"; python /root/code/rllab/scripts/run_experiment_lite.py  --snapshot_mode 'last_best'  --seed '1'  --n_parallel '2'  --exp_name 'CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01'  --log_dir '/tmp/expt'  --use_cloudpickle 'False'  --args_data 'gANjcmxsYWIubWlzYy5pbnN0cnVtZW50ClN0dWJNZXRob2RDYWxsCnEAKYFxAX1xAihYBgAAAF9fYXJnc3EDKGNybGxhYi5taXNjLmluc3RydW1lbnQKU3R1Yk9iamVjdApxBCmBcQV9cQYoWAQAAABhcmdzcQcpWAYAAABrd2FyZ3NxCH1xCShYAwAAAGVudnEKaAQpgXELfXEMKGgHKWgIfXENWAsAAAB3cmFwcGVkX2VudnEOaAQpgXEPfXEQKGgHKWgIfXERKGgKaAQpgXESfXETKGgHKWgIfXEUKFgIAAAAZW52X25hbWVxFVgLAAAAQ2FydFBvbGUtdjBxFlgMAAAAcmVjb3JkX3ZpZGVvcReJWAoAAAByZWNvcmRfbG9ncRiJdVgLAAAAcHJveHlfY2xhc3NxGWNybGxhYi5lbnZzLmd5bV9lbnYKR3ltRW52CnEadWJYDQAAAG5vcm1hbGl6ZV9vYnNxG4l1aBljcmxsYWIuZW52cy5ub3JtYWxpemVkX2VudgpOb3JtYWxpemVkRW52CnEcdWJzaBljc2FuZGJveC5yb2NreS50Zi5lbnZzLmJhc2UKVGZFbnYKcR11YlgGAAAAcG9saWN5cR5oBCmBcR99cSAoaAcpaAh9cSEoWAQAAABuYW1lcSJYCgAAAGNhdF9wb2xpY3lxI1gIAAAAZW52X3NwZWNxJGNybGxhYi5taXNjLmluc3RydW1lbnQKU3R1YkF0dHIKcSUpgXEmfXEnKFgEAAAAX29ianEoaAtYCgAAAF9hdHRyX25hbWVxKVgEAAAAc3BlY3EqdWJYDAAAAGhpZGRlbl9zaXplc3ErXXEsKEtAS0BlWBMAAABoaWRkZW5fbm9ubGluZWFyaXR5cS1jdGVuc29yZmxvdy5weXRob24ub3BzLm1hdGhfb3BzCnRhbmgKcS51aBljc2FuZGJveC5yb2NreS50Zi5wb2xpY2llcy5jYXRlZ29yaWNhbF9tbHBfcG9saWN5CkNhdGVnb3JpY2FsTUxQUG9saWN5CnEvdWJYCAAAAGJhc2VsaW5lcTBoBCmBcTF9cTIoaAcpaAh9cTNoJGglKYFxNH1xNShoKGgLaCloKnVic2gZY3JsbGFiLmJhc2VsaW5lcy5saW5lYXJfZmVhdHVyZV9iYXNlbGluZQpMaW5lYXJGZWF0dXJlQmFzZWxpbmUKcTZ1YlgKAAAAYmF0Y2hfc2l6ZXE3TegDWA8AAABtYXhfcGF0aF9sZW5ndGhxOEvIWAUAAABuX2l0cnE5SxRYCAAAAGRpc2NvdW50cTpHP++uFHrhR65YCQAAAHN0ZXBfc2l6ZXE7Rz+EeuFHrhR7WAoAAABnYWVfbGFtYmRhcTxHP+8KPXCj1wpYDgAAAHNhbXBsZV9iYWNrdXBzcT1LAFgRAAAAa2xfc2FtcGxlX2JhY2t1cHNxPksAWAIAAABxZnE/TlgNAAAAcWZfdXNlX3RhcmdldHFAiFgNAAAAcWZfYmF0Y2hfc2l6ZXFBS0BYCwAAAHFmX21jX3JhdGlvcUJLAFgPAAAAcWZfcmVzaWR1YWxfcGhpcUNLAFgNAAAAbWluX3Bvb2xfc2l6ZXFETegDWAwAAABzY2FsZV9yZXdhcmRxRUc/8AAAAAAAAFgQAAAAcWZfdXBkYXRlc19yYXRpb3FGRz/wAAAAAAAAWBAAAABxcHJvcF9ldGFfb3B0aW9ucUdYBAAAAG9uZXNxSFgQAAAAcmVwbGF5X3Bvb2xfc2l6ZXFJSqCGAQBYEAAAAHJlcGxhY2VtZW50X3Byb2JxSkc/8AAAAAAAAFgLAAAAcWZfYmFzZWxpbmVxS05YEAAAAHFmX2xlYXJuaW5nX3JhdGVxTEc/hHrhR64Ue1gRAAAAYWNfc2FtcGxlX2JhY2t1cHNxTUsAWBIAAABwb2xpY3lfc2FtcGxlX2xhc3RxTohYCQAAAHNhdmVfZnJlcXFPSwFYDAAAAHJlc3RvcmVfYXV0b3FQiFgTAAAAZm9yY2VfYmF0Y2hfc2FtcGxlcnFRiFgKAAAAbl9wYXJhbGxlbHFSSwJYCwAAAHNlcnZlcl9wb3J0cVNYBAAAADgwMThxVFgGAAAAc2NhbGVycVWIWAYAAAB1c2VySWRxVlgYAAAANTlmZjg1ZGNiOWZmNjUzMmQ0YzkyYTA4cVdYBQAAAGV4cElkcVhYGAAAADVhMDc1ZGUxM2NmMDc1MDRmZDBkZmI0NXFZWAkAAAB2YXJpYW50SWRxWlgBAAAAMHFbdWgZY3NhbmRib3gucm9ja3kudGYuYWxnb3MudHJwbwpUUlBPCnFcdWJYBQAAAHRyYWlucV0pfXFedHFfWAgAAABfX2t3YXJnc3FgfXFhdWIu'; sleep 120'

aws s3 cp --recursive /home/ubuntu/rllabpp/data/local/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1 <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1

aws s3 cp /home/ubuntu/user_data.log <insert aws s3 bucket url>/Anil First RL Experiment/CartPole-v0-1000--an-trpo--bc-linear--bhs-32x32--gl-0-97--ksb-0--phn-tanh--phs-64x64--pon-tanh--ss-0-01--s-1/stdout.log

EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die "wget instance-id has failed: $?"`"
aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region us-west-2
} >> /home/ubuntu/user_data.log 2>&1