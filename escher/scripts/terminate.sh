# shell script to terminate the ec2 instance
# sh terminate.sh
# import subprocess
# subprocess.check_call(['sh', 'path to ./terminate.sh'])


instanceId=$(curl http://169.254.169.254/latest/meta-data/instance-id/)
REGION="`wget -qO- http://instance-data/latest/meta-data/placement/availability-zone | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"

aws ec2 terminate-instances --instance-ids $instanceId --region $REGION