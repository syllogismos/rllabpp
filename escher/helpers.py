import subprocess


def terminate_running_ec2_instance():
    try:
        subprocess.check_call(['sh', '/home/ubuntu/rllabpp/escher/scripts/terminate.sh'])
    except:
        print("Failed to terminate ec2 instance")