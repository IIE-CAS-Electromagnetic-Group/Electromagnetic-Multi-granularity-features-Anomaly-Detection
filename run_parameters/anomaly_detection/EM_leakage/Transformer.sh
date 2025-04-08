--task_name=forecast_and_anomaly_detection
--is_training=1
--root_path=/home/weifeng/Work/Workspace/Data/对讲机信号/intercom_prediction
--model_id=IntercomPred_60_30
--model=Transformer
--data=IntercomPred
--target=416.08637
--freq=s
--features=M
--seq_len=60
--label_len=15
--pred_len=30
--e_layers=2
--d_layers=1
--factor=3
--enc_in=27
--dec_in=27
--c_out=27
--d_ff=32
--des='Exp'
--itr=1
--batch_size=64
--anomaly_ratio=7
--patience=5
--train_epochs=20


--task_name=anomaly_detection
--is_training=1
--root_path=G:/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage
--model_id=SignalSeg_20
--model=Transformer
--data=SignalFineSeg
--target=1.508925
--freq=s
--features=M
--seq_len=20
--label_len=1
--pred_len=1
--e_layers=2
--d_layers=1
--factor=3
--enc_in=59
--dec_in=59
--c_out=59
--d_model=128
--d_ff=128
--des='Exp'
--itr=1
--batch_size=64
--anomaly_ratio=1
--patience=4
--train_epochs=10



--task_name=anomaly_detection
--is_training=2
--root_path=/media/weifeng/移动硬盘/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage
--model_id=SignalSeg_20
--model=Transformer
--data=SignalFineSeg
--target=1.508925
--freq=s
--features=M
--seq_len=20
--label_len=1
--pred_len=1
--e_layers=2
--d_layers=1
--factor=3
--enc_in=50
--dec_in=50
--c_out=50
--d_model=128
--d_ff=128
--des='Exp'
--itr=1
--batch_size=128
--anomaly_ratio=1
--patience=4
--train_epochs=10



#附加信息

--task_name=anomaly_detection
--is_training=2
--root_path=/media/weifeng/移动硬盘/Work/Workspace/Data/模型输入输出数据或其它处理结果/电磁泄漏信号/模型训练数据/EM_leakage
--model_id=SignalSeg_30
--model=Transformer
--data=SignalFineSeg
--target=1.508925
--freq=s
--features=M
--seq_len=30
--label_len=1
--pred_len=1
--e_layers=2
--d_layers=1
--factor=3
--enc_in=52
--dec_in=52
--c_out=52
--d_model=128
--d_ff=128
--des='Exp'
--itr=1
--batch_size=256
--anomaly_ratio=1
--patience=4
--train_epochs=15




