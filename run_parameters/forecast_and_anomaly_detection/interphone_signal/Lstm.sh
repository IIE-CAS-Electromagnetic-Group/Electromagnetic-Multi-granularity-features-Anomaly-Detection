--task_name=forecast_and_anomaly_detection
--is_training=2
--root_path=/media/weifeng/移动硬盘/Work/Workspace/Data/模型输入输出数据或其它处理结果/对讲机信号/模型训练数据/interphone_signal
--model_id=InterphonePred_10_1
--model=Lstm
--data=SignalCoarsePred
--target=communication_num
--freq=t
--features=M
--seq_len=10
--label_len=1
--pred_len=1
--e_layers=2
--d_layers=1
--factor=3
--enc_in=86
--dec_in=86
--c_out=86
--d_model=128
--d_ff=128
--des='Exp'
--itr=1
--batch_size=128
--anomaly_ratio=1
--patience=4
--train_epochs=3