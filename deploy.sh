#!/bin/bash

KEY="ec2-web-instance.pem"
WORKER_FILE="worker/worker.py"
RECIPE_FILE="com.fl.Worker-1.0.0.json"

while read -r ip worker_num; do
    if [ -z "$ip" ]; then continue; fi
    
    # reset, lefrt from p1
    # ssh -n -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$ip" \
    # "crontab -r; sudo pkill -f worker.py"
    
    echo "========================================"
    echo "Deploying Greengrass Component to IP: $ip"
    
    # folder
    ssh -n -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$ip" \
    "mkdir -p ~/greengrassv2/recipes ~/greengrassv2/artifacts/com.fl.Worker/1.0.0"
    
    # worker.py
    scp -i "$KEY" -o StrictHostKeyChecking=no "$WORKER_FILE" \
    ubuntu@"$ip":~/greengrassv2/artifacts/com.fl.Worker/1.0.0/
    
    # the Recipe
    echo "Recipe"
    scp -i "$KEY" -o StrictHostKeyChecking=no "$RECIPE_FILE" \
    ubuntu@"$ip":~/greengrassv2/recipes/
    
    # restart the component
    ssh -n -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$ip" \
    'sudo /greengrass/v2/bin/greengrass-cli deployment create \
    --recipeDir ~/greengrassv2/recipes \
    --artifactDir ~/greengrassv2/artifacts \
    --merge "com.fl.Worker=1.0.0"'
    
    # ssh -n -i "$KEY" -o StrictHostKeyChecking=no ubuntu@"$ip" \
    # 'sudo /greengrass/v2/bin/greengrass-cli deployment create --remove "com.fl.Worker" && \
    #  sudo /greengrass/v2/bin/greengrass-cli deployment create \
    # --recipeDir ~/greengrassv2/recipes \
    # --artifactDir ~/greengrassv2/artifacts \
    # --merge "com.fl.Worker=1.0.0"'
    
    echo "Finished $ip"
    
done < ips.txt

echo "========================================"
echo "ALL DONE"