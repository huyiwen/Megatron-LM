#!/bin/bash

SCRIPT=/data/Megatron-LM/submit-job.sh

while true; do
    # 获取包含train的pod信息
    pod_status=$(kubectl get pods -o wide | grep train | grep -v Init)

    # 检查是否有非Running状态的pod
    if echo "$pod_status" | grep -v "Running" > /dev/null; then
        echo "发现非Running状态的pod:"
        echo "$pod_status"

        # 执行你的bash脚本
        bash $SCRIPT
        curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data "{\"appToken\": \"AT_6x1rUKLWJsd3DGyvm7NNxpI3GNr7bEN5\", \"content\": \"resubmit ${pod_status:-Empty}\", \"topicIds\": [37328]}"
        echo "脚本执行完毕,继续监控..."
        sleep 240
    fi

    error_logs=$(kubectl get pods -o wide | grep train | awk '{print $1}' | xargs -I {} sh -c 'kubectl logs {} | tail -50 | grep Error' 2>/dev/null)

    if [ ! -z "$error_logs" ]; then
        echo "发现Error日志:"
        echo "$error_logs"

        # 执行你的bash脚本
        bash $SCRIPT
        curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data "{\"appToken\": \"AT_6x1rUKLWJsd3DGyvm7NNxpI3GNr7bEN5\", \"content\": \"resubmit $error_logs\", \"topicIds\": [37328]}"
        echo "脚本执行完毕,继续监控..."
        sleep 240
    fi

    # 休眠5秒后继续检查
    sleep 5
done